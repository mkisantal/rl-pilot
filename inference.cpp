#include "inference.h"

#include <opencv2/core/core.hpp>

#include "tensorflow/core/framework/graph.pb.h"   // TODO: remove unused headers
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/logging.h"

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


#include <iostream>



using tensorflow::Tensor;
using tensorflow::Status;


int Inference::Init(std::unique_ptr<tensorflow::Session>* session){

	/* loading TF graph from .pb file, initializing session */

    tensorflow::GraphDef graph_def;

    Status load_graph_status =
    	tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path_, &graph_def);


    if (!load_graph_status.ok()) {
    	std::cout << "Failed to load graph from " << graph_path_ << std::endl;
    	std::cout << load_graph_status << std::endl;
        return -1;
    }

    if (!load_graph_status.ok()) {
        //LOG(ERROR) << load_graph_status;
        std::cout << load_graph_status << std::endl;
        return -1;
    }

    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        std::cout << session_create_status << std::endl;
        return -1;
    }


    return 0; // maybe do first inference on fake data just to speed up subsequent calls
    
}

void Inference::PreprocessImage(cv::Mat& input_image, cv::Mat& preprocessed_image){
	
	/* normalize images to range [-1, 1] */

	input_image.convertTo(preprocessed_image,CV_32FC3);
	double alpha = 1.0/127.5;
	double beta = -1.0;
	preprocessed_image.convertTo(preprocessed_image,-1,alpha,beta);
}


void Inference::TensorInitializer(tensorflow::Tensor& tnzr, const cv::Mat& input_cv_mat){

	/* initializing tensor from OpenCV matrix */

	int depth = input_cv_mat.channels();
	const int input_rows = input_cv_mat.rows;
	const int input_cols = input_cv_mat.cols;

	const float * source_data = (float*) input_cv_mat.data;

	if (depth == 1){

		// copy matrix
		auto tnzr_mapped = tnzr.tensor<float, 2>();

		for (int y = 0; y < input_rows; ++y) {
	    	const float* source_row = source_data + (y * input_cols * depth);
	    	for (int x = 0; x < input_cols; ++x) {
	    		const float* source_value = source_row + (x * depth);
				tnzr_mapped(0, x) = *source_value;
	    	}
		}

	} else {

		// copy RGB input image
		auto tnzr_mapped = tnzr.tensor<float, 4>();

		for (int y = 0; y < input_rows; ++y) {
    		const float* source_row = source_data + (y * input_cols * depth);
    		for (int x = 0; x < input_cols; ++x) {
    			const float* source_pixel = source_row + (x * depth);
    			for (int c = 0; c < depth; ++c) {
  					const float* source_value = source_pixel + c;
					tnzr_mapped(0, y, x, c) = *source_value;
      			}
    		}
		}
	}
}

void Inference::TensorInitializer(tensorflow::Tensor& tnzr, const float data){
	
	/* initializing tensor with scalar float data */

    auto tnzr_mapped = tnzr.tensor<float, 2>();
    tnzr_mapped(0, 0) = data;
}


void Inference::CollectInputs(std::vector<std::pair<std::string, tensorflow::Tensor>>& input_feed){

	/* collecting the input feed for inference*/

	// vector has to be empty!

	// input image
	PreprocessImage(latest_frame_, preprocessed_frame_);
	TensorInitializer(image_input_tensor_, preprocessed_frame_);


	// other inputs
	TensorInitializer(velocity_state_tensor_, latest_velocity_);
	TensorInitializer(heading_error_tensor_, latest_hdg_error_);
	TensorInitializer(lstm_state_tensor_, init_lstm_state_);
	TensorInitializer(prev_action_tensor_, latest_action_);
	TensorInitializer(prev_reward_tensor_, latest_reward_);

	// collect input feed to vector
	input_feed.push_back(
		std::make_pair(image_placeholder_, image_input_tensor_));
    input_feed.push_back(
    	std::make_pair(velocity_state_placeholder_, velocity_state_tensor_));
    input_feed.push_back(
    	std::make_pair(heading_error_placeholder_, heading_error_tensor_));
    input_feed.push_back(
    	std::make_pair(lstm_state_placeholder_, lstm_state_tensor_));
    input_feed.push_back(
    	std::make_pair(prev_reward_placeholder_, prev_reward_tensor_));
    input_feed.push_back(
    	std::make_pair(prev_action_placeholder_, prev_action_tensor_));
}



int Inference::Run(std::unique_ptr<tensorflow::Session>& session){
	
	/* Collecting inputs and running inference on the loaded graph. */

	std::vector<std::pair<std::string, tensorflow::Tensor>> input_feed;
	CollectInputs(input_feed);
    
    // run inference on the graph
    Status run_status =
    	session->Run(input_feed,
    				{action_distribution_node_, value_prediction_node_, lstm_state_out_node_},
                    {}, &output_tensors_);

    if (!run_status.ok()) {
        //tensorflow::LOG(ERROR) << "Running model failed: " << run_status;
        std::cout << run_status << std::endl;
        return -1;
    }

    std::cout << output_tensors_[0].DebugString() << std::endl;	// DEBUG

    return 0;
}

void Inference::NewImageInput(cv::Mat new_frame){
	latest_frame_ = new_frame;
}