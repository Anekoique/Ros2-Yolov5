#include "image_base64.hpp"
#include <vector>
#include "opencv2/opencv.hpp"

static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

const std::string ImageBase64::encode(const unsigned char* Data, int DataByte)
{
    //编码表  
    const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    //返回值  
    std::string str_encode;
    unsigned char tmp[4] = { 0 };
    int line_length = 0;
    for (int i = 0; i < (int)(DataByte / 3); i++)
    {
        tmp[1] = *Data++;
        tmp[2] = *Data++;
        tmp[3] = *Data++;
        str_encode += EncodeTable[tmp[1] >> 2];
        str_encode += EncodeTable[((tmp[1] << 4) | (tmp[2] >> 4)) & 0x3F];
        str_encode += EncodeTable[((tmp[2] << 2) | (tmp[3] >> 6)) & 0x3F];
        str_encode += EncodeTable[tmp[3] & 0x3F];
        if (line_length += 4, line_length == 76)
        {
            str_encode += "\r\n"; 
            line_length = 0;
        }
    }
    //对剩余数据进行编码  
    int mod = DataByte % 3;
    if (mod == 1)
    {
        tmp[1] = *Data++;
        str_encode += EncodeTable[(tmp[1] & 0xFC) >> 2];
        str_encode += EncodeTable[((tmp[1] & 0x03) << 4)];
        str_encode += "==";
    }
    else if (mod == 2)
    {
        tmp[1] = *Data++;
        tmp[2] = *Data++;
        str_encode += EncodeTable[(tmp[1] & 0xFC) >> 2];
        str_encode += EncodeTable[((tmp[1] & 0x03) << 4) | ((tmp[2] & 0xF0) >> 4)];
        str_encode += EncodeTable[((tmp[2] & 0x0F) << 2)];
        str_encode += "=";
    }

    return str_encode;
}

const std::string ImageBase64::decode(const char* Data, int DataByte, int& OutByte)
{
    //解码表  
    const char DecodeTable[] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        62, // '+'  
        0, 0, 0,
        63, // '/'  
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'  
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'  
        0, 0, 0, 0, 0, 0,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'  
    };
    //返回值  
    std::string str_decode;
    int n_value = 0;
    int i = 0;
    while (i < DataByte)
    {
        if (*Data != '\r' && *Data != '\n')
        {
            n_value = DecodeTable[*Data++] << 18;
            n_value += DecodeTable[*Data++] << 12;
            str_decode += (n_value & 0x00FF0000) >> 16;
            OutByte++;
            if (*Data != '=')
            {
                n_value += DecodeTable[*Data++] << 6;
                str_decode += (n_value & 0x0000FF00) >> 8;
                OutByte++;
                if (*Data != '=')
                {
                    n_value += DecodeTable[*Data++];
                    str_decode += n_value & 0x000000FF;
                    OutByte++;
                }
            }
            i += 4;
        }
        else// 回车换行,跳过  
        {
            Data++;
            i++;
        }
    }
    return str_decode;
}

std::string ImageBase64::file2base64(const std::string& file_path)
{
    cv::Mat img = cv::imread(file_path);
    if (img.empty())
    {
        return "";
    }
    std::vector<uchar> vec_img;                               //Mat 图片数据转换为vector<uchar>  
    std::vector<int> vec_compression_params;
    vec_compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    vec_compression_params.push_back(90);
    cv::imencode(".jpg", img, vec_img, vec_compression_params);
    // base64;
    std::string img_base64 = encode(vec_img.data(), vec_img.size());
    return img_base64;
}

