#ifndef TESTFACEAPI_TESTFACEAPI_IMAGE_BASE64_H
#define TESTFACEAPI_TESTFACEAPI_IMAGE_BASE64_H
#include <string>
// 对图片进行base64编码解码类
class ImageBase64
{
public:

    /*编码
    DataByte
    [in]输入的数据长度,以字节为单位
    */
    static const std::string encode(const unsigned char* Data, int DataByte);
    /*解码
    DataByte
    [in]输入的数据长度,以字节为单位
    OutByte
    [out]输出的数据长度,以字节为单位,请不要通过返回值计算
    输出数据的长度
    */
    static const std::string decode(const char* Data, int DataByte, int& OutByte);
    // 传入图片地址进行base64编码示例
    static std::string file2base64(const std::string& file_path);
};
#endif

