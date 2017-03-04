#include <fstream>
#include <string>
#include <regex>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Eigen/Eigen>
#include <torch/Exception.h>

DEFINE_string(poses, "poses.csv", "pose file to be processed");
DEFINE_string(out, "new_poses.csv", "output file for processed poses");
DEFINE_bool(euler, false, "output rotation in Euler angles");

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Processing " << FLAGS_poses << "..." << std::endl;
  LOG(INFO) << "Writing to " << FLAGS_out << "..." << std::endl;

  Eigen::Matrix4d Tir;
  Tir.block<1, 4>(0, 0) = Eigen::Vector4d( 0.999961900,  0.008251156,  0.002856259,  0.026387510);
  Tir.block<1, 4>(1, 0) = Eigen::Vector4d(-0.008240786,  0.999959500, -0.003623658,  0.000004579);
  Tir.block<1, 4>(2, 0) = Eigen::Vector4d(-0.002886043,  0.003599982,  0.999989400,  0.003246200);
  Tir.block<1, 4>(3, 0) = Eigen::Vector4d( 0.000000000,  0.000000000,  0.000000000,  1.000000000);

  std::ifstream fin(FLAGS_poses);
  std::ofstream fout(FLAGS_out);
  std::string line;

  while (std::getline(fin, line))
  {
    Eigen::Matrix4d Tiw;
    std::stringstream tokenizer(line);
    std::string token;

    for (int i = 0; i < 16; ++i)
    {
      TORCH_ASSERT(std::getline(tokenizer, token, ','), "invalid pose file");
      Tiw.data()[i] = std::stod(token);
    }

    const Eigen::Matrix4d Twr = Tiw.inverse() * Tir;

    if (FLAGS_euler)
    {
      Eigen::VectorXd new_pose(6);
      new_pose.segment<3>(0) = Twr.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
      new_pose.segment<3>(3) = Twr.block<3, 1>(0, 3);

      for (int i = 0; i < 5; ++i)
      {
        fout << new_pose[i] << ", ";
      }

      fout << new_pose[5] << std::endl;;
    }
    else
    {
      Eigen::VectorXd new_pose(7);
      const Eigen::Quaterniond quaternion(Twr.block<3, 3>(0, 0));
      new_pose.segment<4>(0) = quaternion.coeffs();
      new_pose.segment<3>(4) = Twr.block<3, 1>(0, 3);

      for (int i = 0; i < 6; ++i)
      {
        fout << new_pose[i] << ", ";
      }

      fout << new_pose[6] << std::endl;;
    }
  }

  fin.close();
  fout.close();
  LOG(INFO) << "Success" << std::endl;
  return 0;
}