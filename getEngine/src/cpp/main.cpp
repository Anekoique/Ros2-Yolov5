#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	Model model("model/onnx/circle.onnx");
	if (!model.build)
}
