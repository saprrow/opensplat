#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include <cxxopts.hpp>
#include <chrono> 


#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("keep-crs", "Retain the project input's coordinate reference system")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))

        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odm project path]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }


    const std::string projectRoot = result["input"].as<std::string>();
    const std::string outputScene = result["output"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>(); 
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0;
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    const bool keepCrs = result.count("keep-crs") > 0;
    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f);
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int refineEvery = result["refine-every"].as<int>();
    const int warmupLength = result["warmup-length"].as<int>();
    const int resetAlphaEvery = result["reset-alpha-every"].as<int>();
    const float densifyGradThresh = result["densify-grad-thresh"].as<float>();
    const float densifySizeThresh = result["densify-size-thresh"].as<float>();
    const int stopScreenSizeAt = result["stop-screen-size-at"].as<int>();
    const float splitScreenSize = result["split-screen-size"].as<float>();

    torch::Device device = torch::kCPU;
    int displayStep = 10;

    if (torch::hasCUDA() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::hasMPS() && result.count("cpu") == 0) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    }else{
        std::cout << "Using CPU" << std::endl;
        displayStep = 1;
    }

#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    visualizer.Initialize(numIters);
#endif

    try{
        auto start_time = std::chrono::high_resolution_clock::now();

        InputData inputData = inputDataFromX(projectRoot);


        OptimizedImageLoader loader(1024*4); // 限制使用1GB内存
        loader.loadImages(inputData.cameras, downScaleFactor);

        // parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
        //     cam.loadImage(downScaleFactor);
        // });

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                    refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                    numIters, keepCrs,
                    device);

        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        int imageSize = -1;
        for (size_t step = 1; step <= numIters; step++){
            Camera& cam = cams[ camsIter.next() ];

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, ssimWeight);
            mainLoss.backward();
            
            if (step % displayStep == 0) std::cout << "Step " << step << ": " << mainLoss.item<float>() << std::endl;

            model.optimizersStep();
            model.schedulersStep(step);
            model.afterTrain(step);

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.save((p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string()));
            }

            if (!valRender.empty() && step % 10 == 0){
                torch::Tensor rgb = model.forward(*valCam, step);
                cv::Mat image = tensorToImage(rgb.detach().cpu());
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);
            }

#ifdef USE_VISUALIZATION
            visualizer.SetInitialGaussianNum(inputData.points.xyz.size(0));
            visualizer.SetLoss(step, mainLoss.item<float>());
            visualizer.SetGaussians(model.means, model.scales, model.featuresDc,
                                    model.opacities);
            visualizer.SetImage(rgb, gt);
            visualizer.Draw();
#endif
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene);
        // model.saveDebugPly("debug.ply");

        // Validate
        if (valCam != nullptr){
            torch::Tensor rgb = model.forward(*valCam, numIters);
            torch::Tensor gt = valCam->getImage(model.getDownscaleFactor(numIters)).to(device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, ssimWeight).item<float>() << std::endl; 
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        int hours = duration / 3600;
        int minutes = (duration % 3600) / 60;
        int seconds = duration % 60;

        std::cout << "processed " << cams.size() << " images for " << numIters << "steps, cost time: " 
                      << std::setfill('0') << std::setw(2) << hours << ":"
                      << std::setfill('0') << std::setw(2) << minutes << ":"
                      << std::setfill('0') << std::setw(2) << seconds << std::endl;


    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
