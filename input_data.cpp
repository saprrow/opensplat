#include <filesystem>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"


namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor) {
    if (image.numel()) throw std::runtime_error("loadImage already called");

    // 1. 直接读取时考虑缩放
    float totalScale = 1.0f;
    if (downscaleFactor > 1.0f) {
        totalScale = 1.0f / downscaleFactor;
    }

    // 2. 读取并缩放图像
    cv::Mat cImg;
    if (totalScale != 1.0f) {
        // 先读取原图
        cv::Mat temp = cv::imread(filePath);
        if (temp.empty()) {
            throw std::runtime_error("Failed to load image: " + filePath);
        }
        
        // 计算目标尺寸
        int target_width = static_cast<int>(temp.cols * totalScale);
        int target_height = static_cast<int>(temp.rows * totalScale);
        
        // 缩放
        cv::resize(temp, cImg, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
        temp.release();  // 立即释放原图
    } else {
        cImg = cv::imread(filePath);
        if (cImg.empty()) {
            throw std::runtime_error("Failed to load image: " + filePath);
        }
    }

    // 更新相机参数
    float rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    fx *= rescaleF;
    fy *= rescaleF;
    cx *= rescaleF;
    cy *= rescaleF;

    K = getIntrinsicsMatrix();

    // 3. 处理畸变校正
    if (hasDistortionParameters()) {
        std::vector<float> distCoeffs = undistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(K);
        cv::Rect roi;
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, 
            cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);
        
        // 在原地进行畸变校正
        cv::Mat undistorted;
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        cImg.release();  // 立即释放原图
        
        // 转换为tensor
        image = imageToTensor(undistorted);
        undistorted.release();
        K = floatNxNMatToTensor(newK);
        
        // 如果需要裁剪ROI
        if (roi.x != 0 || roi.y != 0 || 
            roi.width != image.size(1) || roi.height != image.size(0)) {
            image = image.index({
                Slice(roi.y, roi.y + roi.height), 
                Slice(roi.x, roi.x + roi.width), 
                Slice()
            }).clone();  // 确保内存连续
        }
    } else {
        image = imageToTensor(cImg);
        cImg.release();
    }

    // 更新最终参数
    height = image.size(0);
    width = image.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
}



torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> Camera::undistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}


void InputData::saveCameras(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.filePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)}).squeeze();
        
        // Flip z and y
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));

        if (keepCrs) T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}

// 实现构造函数
OptimizedImageLoader::OptimizedImageLoader(size_t maxMemoryMB) 
    : maxMemoryBytes(maxMemoryMB * 1024 * 1024) {}

// 实现loadImages方法
void OptimizedImageLoader::loadImages(std::vector<Camera>& cameras, float downscaleFactor) {
    if (cameras.empty()) return;

    size_t estimatedMemPerImage = estimateImageMemory(cameras[0]);
    size_t batchSize = std::max(
        size_t(1),
        maxMemoryBytes / (estimatedMemPerImage * 3)
    );

    for(size_t i = 0; i < cameras.size(); i += batchSize) {
        size_t currentBatch = std::min(batchSize, cameras.size() - i);
        
        parallel_for(
            cameras.begin() + i,
            cameras.begin() + i + currentBatch,
            [&downscaleFactor](Camera &cam){
                cam.loadImage(downscaleFactor);
            }
        );

        std::cout << "Processed " << std::min(i + batchSize, cameras.size()) 
                 << "/" << cameras.size() << " images\r" << std::flush;
    }
    std::cout << std::endl;
}

// 实现estimateImageMemory方法
size_t OptimizedImageLoader::estimateImageMemory(const Camera& camera) {
    return camera.width * camera.height * 3 * sizeof(float);
}