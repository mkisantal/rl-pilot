#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <string>
#include <random>

#include <opencv2/core/core.hpp>

#include "tensorflow/core/public/session.h"  // TODO: remove unused headers
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


class Inference{

public:

	int Init(std::unique_ptr<tensorflow::Session>* session);
	int Run(std::unique_ptr<tensorflow::Session>& session);
	void NewImageInput(cv::Mat new_frame);

private:

	const int img_width_ = 84;
	const int img_height_ = 84;

	const std::string graph_path_ = "model/output_graph.pb";

	// input data
	cv::Mat latest_frame_ = cv::Mat::zeros(img_width_,img_height_,CV_8UC3);
	cv::Mat preprocessed_frame_ = cv::Mat::zeros(img_width_,img_height_,CV_32FC3);
	float latest_velocity_ = 0.0;
	float latest_hdg_error_ = 0.0;
	cv::Mat init_lstm_state_ = cv::Mat::zeros(1,512,CV_32FC1);
	cv::Mat latest_action_ = cv::Mat::zeros(1,3,CV_32FC1);
	float latest_reward_ = 0.0f;

	// input placeholders
	const std::string image_placeholder_ = "global/input_image";
    tensorflow::Tensor image_input_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, img_width_, img_height_, 3}));

    const std::string velocity_state_placeholder_ = "global/velocity_state";
    tensorflow::Tensor velocity_state_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));

    const std::string heading_error_placeholder_ = "global/heading_error_input";
    tensorflow::Tensor heading_error_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));

    const std::string lstm_state_placeholder_ = "global/Placeholder";
    tensorflow::Tensor lstm_state_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 512}));

    const std::string prev_action_placeholder_ = "global/previous_action";
    tensorflow::Tensor prev_action_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 3}));

    const std::string prev_reward_placeholder_ = "global/previous_reward";
    tensorflow::Tensor prev_reward_tensor_ = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1}));

    // output names
    const std::string action_distribution_node_ = "global/policy_out/Softmax";
    const std::string value_prediction_node_ = "global/value_out/MatMul";
    const std::string lstm_state_out_node_ = "global/Slice_1";
    std::vector<tensorflow::Tensor> output_tensors_;

    // random number generation
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> uniform_distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);


    void CollectInputs(std::vector<std::pair<std::string, tensorflow::Tensor>>& input_feed);
	void TensorInitializer(tensorflow::Tensor& tnzr, const cv::Mat& input_cv_mat);
	void TensorInitializer(tensorflow::Tensor& tnzr, const float data);
    void PreprocessImage(cv::Mat& input_image, cv::Mat& preprocessed_image);
    int GreedyActionSelection(const tensorflow::Tensor& action_distribution);
    int StochasticActionSelection(const tensorflow::Tensor& action_distribution);
    void UpdateOnehotPrevAction(int action);
};

#endif // INFERENCE_H_