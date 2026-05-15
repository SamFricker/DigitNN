#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#include "stb_image_resize2.h"



struct MNISTData {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
    uint32_t num_images, rows, cols;
};

uint32_t readBigEndianUInt32(std::ifstream &ifs) {
    uint8_t bytes[4];
    ifs.read((char*)bytes, 4);
    return (uint32_t(bytes[0]) << 24)
         | (uint32_t(bytes[1]) << 16)
         | (uint32_t(bytes[2]) <<  8)
         |  uint32_t(bytes[3]);
}

MNISTData loadMNIST(const std::string &image_path, const std::string &label_path) {
    MNISTData data;
    std::ifstream img_file(image_path, std::ios::binary);
    if (!img_file) { std::cerr<<"Failed to open "<<image_path<<"\n"; exit(1); }
    if (readBigEndianUInt32(img_file)!=2051) { std::cerr<<"Bad magic\n"; exit(1); }
    data.num_images = readBigEndianUInt32(img_file);
    data.rows       = readBigEndianUInt32(img_file);
    data.cols       = readBigEndianUInt32(img_file);
    img_file.seekg(16, std::ios::beg);

    std::ifstream lbl_file(label_path, std::ios::binary);
    if (!lbl_file) { std::cerr<<"Failed to open "<<label_path<<"\n"; exit(1); }
    if (readBigEndianUInt32(lbl_file)!=2049) { std::cerr<<"Bad magic\n"; exit(1); }
    uint32_t num_labels = readBigEndianUInt32(lbl_file);
    if (num_labels!=data.num_images) { std::cerr<<"Count mismatch\n"; exit(1); }
    lbl_file.seekg(8, std::ios::beg);

    data.images.resize(data.num_images, std::vector<uint8_t>(data.rows*data.cols));
    for (uint32_t i=0; i<data.num_images; ++i)
        img_file.read((char*)data.images[i].data(), data.rows*data.cols);

    data.labels.resize(data.num_images);
    lbl_file.read((char*)data.labels.data(), data.num_images);

    return data;
}

std::vector<std::vector<float>> normaliseImages(const std::vector<std::vector<uint8_t>>& images) {
    std::vector<std::vector<float>> normalised;
    normalised.reserve(images.size());
    for (auto &img : images) {
        std::vector<float> v; v.reserve(img.size());
        for (auto px : img) v.push_back(px/255.0f - 0.5f);
        normalised.push_back(std::move(v));
    }
    return normalised;
}

std::vector<std::vector<float>> oneHotLabels(const std::vector<uint8_t>& labels) {
    std::vector<std::vector<float>> encoded;
    encoded.reserve(labels.size());
    for (auto label : labels) {
        std::vector<float> vec(10, 0.1f);
        vec[label] = 0.9f;
        encoded.push_back(std::move(vec));
    }
    return encoded;
}

void xavierInit2D(std::vector<std::vector<float>>& W, int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f/(fan_in+fan_out));
    for (auto &row : W)
        for (auto &x : row)
            x = ((float)rand()/RAND_MAX) * 2*limit - limit;
}

void initBias1D(std::vector<float>& b, float val=-0.1f) {
    for (auto &x : b) x = val;
}

std::vector<float> sigmoid1D(const std::vector<float>& nodes){
    std::vector<float> v(nodes.size(),0.0f);
    for (int i=0; i<(int)nodes.size(); ++i) {
        if (nodes[i] > 50)        v[i]=1.0f;
        else if (nodes[i] < -50)  v[i]=0.0f;
        else                      v[i]=1.0f/(1.0f+expf(-nodes[i]));
    }
    return v;
}

std::vector<float> multiplyMatrices(
    const std::vector<float>& prevLayer,
    const std::vector<std::vector<float>>& weights)
{
    std::vector<float> result(weights.size(), 0.0f);
    for (int i=0; i<(int)weights.size(); ++i)
        for (int j=0; j<(int)weights[i].size(); ++j)
            result[i] += prevLayer[j] * weights[i][j];
    return result;
}

class NeuralNetwork {
public:
    static constexpr int input_size=784, hidden1_size=64, hidden2_size=16, output_size=10;

    std::vector<float> hidden1, hidden2, output;
    std::vector<float> zhidden1, zhidden2, zoutput;
    std::vector<float> delta_output, delta_hidden2, delta_hidden1;

    std::vector<std::vector<float>> w0_1, w1_2, w2_3;
    std::vector<float> b1, b2, b3;

    std::vector<std::vector<float>> grad_w0_1, grad_w1_2, grad_w2_3;
    std::vector<std::vector<float>> vel_w0_1, vel_w1_2, vel_w2_3;
    std::vector<float> vel_b1, vel_b2, vel_b3;

    float eta=0.001f, momentum=0.9f, clip_th=1.0f;
    float correct=0;

    NeuralNetwork()
      : hidden1(hidden1_size), hidden2(hidden2_size), output(output_size),
        zhidden1(hidden1_size), zhidden2(hidden2_size), zoutput(output_size),
        delta_output(output_size), delta_hidden2(hidden2_size), delta_hidden1(hidden1_size),
        w0_1(hidden1_size, std::vector<float>(input_size)),
        w1_2(hidden2_size, std::vector<float>(hidden1_size)),
        w2_3(output_size, std::vector<float>(hidden2_size)),
        b1(hidden1_size), b2(hidden2_size), b3(output_size),
        grad_w0_1(hidden1_size, std::vector<float>(input_size)),
        grad_w1_2(hidden2_size, std::vector<float>(hidden1_size)),
        grad_w2_3(output_size, std::vector<float>(hidden2_size)),
        vel_w0_1(hidden1_size, std::vector<float>(input_size,0.0f)),
        vel_w1_2(hidden2_size, std::vector<float>(hidden1_size,0.0f)),
        vel_w2_3(output_size,   std::vector<float>(hidden2_size,0.0f)),
        vel_b1(hidden1_size,0.0f), vel_b2(hidden2_size,0.0f), vel_b3(output_size,0.0f)
    {}

    void randomiseParameters(){
        xavierInit2D(w0_1, input_size, hidden1_size);
        xavierInit2D(w1_2, hidden1_size, hidden2_size);
        xavierInit2D(w2_3, hidden2_size, output_size);
        initBias1D(b1,-0.1f);
        initBias1D(b2,-0.1f);
        initBias1D(b3,-0.1f);
    }

    std::string filepath = "C:\\Users\\samfr\\OneDrive\\Desktop\\Digit NN\\parameters\\perameters.txt";

    void loadParameters (){
        std::ifstream inFile(filepath);
        if (!inFile) {
            std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        } else {
            std::string line;
            int lineCount = 0;
            while (std::getline(inFile, line)) {
                std::istringstream iss(line);
                float value;
                int column = 0;

                if (lineCount < 64) {
                    while (iss >> value) {
                        w0_1[lineCount][column] = value;
                        column++;
                    }
                } else if ((64 <= lineCount) && (lineCount < 80)){
                    while (iss >> value) {
                        w1_2[lineCount - 64][column] = value;
                        column++;
                    }
                } else if ((80 <= lineCount) && (lineCount < 90)){
                    while (iss >> value) {
                        w2_3[lineCount - 80][column] = value;
                        column++;
                    }
                } else if (lineCount == 90){
                    while (iss >> value) {
                        b1[column] = value;
                        column++;
                    }
                } else if (lineCount == 91){
                    while (iss >> value) {
                        b2[column] = value;
                        column++;
                    }
                } else if (lineCount == 92){
                    while (iss >> value) {
                        b3[column] = value;
                        column++;
                    }
                }

                lineCount++;
            }
        }
    }

    void saveParameters(){
        std::ofstream outFile(filepath);
        if (!outFile) {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        } else {
            for(int i = 0; i < w0_1.size(); i++){
                for(int j = 0; j < w0_1[i].size(); j++){
                    outFile << w0_1[i][j] << " ";
                }
                outFile << "\n";
            }
            for(int i = 0; i < w1_2.size(); i++){
                for(int j = 0; j < w1_2[i].size(); j++){
                    outFile << w1_2[i][j] << " ";
                }
                outFile << "\n";
            }
            for(int i = 0; i < w2_3.size(); i++){
                for(int j = 0; j < w2_3[i].size(); j++){
                    outFile << w2_3[i][j] << " ";
                }
                outFile << "\n";
            }
            for(int i = 0; i < b1.size(); i++){
                outFile << b1[i] << " ";
            }
            outFile << "\n";
            for(int i = 0; i < b2.size(); i++){
                outFile << b2[i] << " ";
            }
            outFile << "\n";
            for(int i = 0; i < b3.size(); i++){
                outFile << b3[i] << " ";
            }

            outFile.close();
        }
    }

    void forwardPass(const std::vector<float>& image){
        zhidden1 = multiplyMatrices(image, w0_1);
        for(int i=0;i<hidden1_size;++i) zhidden1[i] += b1[i];
        hidden1  = sigmoid1D(zhidden1);

        zhidden2 = multiplyMatrices(hidden1, w1_2);
        for(int i=0;i<hidden2_size;++i) zhidden2[i] += b2[i];
        hidden2  = sigmoid1D(zhidden2);

        zoutput  = multiplyMatrices(hidden2, w2_3);
        for(int i=0;i<output_size;++i) zoutput[i] += b3[i];
        output   = sigmoid1D(zoutput);
    }

    void backProp(const std::vector<float>& input, const std::vector<float>& label){
        for(int i=0;i<output_size;++i)
            delta_output[i] = (output[i]-label[i]) * output[i] * (1.0f-output[i]);

        std::fill(delta_hidden2.begin(), delta_hidden2.end(), 0.0f);
        for(int j=0;j<hidden2_size;++j){
            float sum=0;
            for(int k=0;k<output_size;++k)
                sum += delta_output[k] * w2_3[k][j];
            delta_hidden2[j] = sum * hidden2[j] * (1.0f-hidden2[j]);
        }

        std::fill(delta_hidden1.begin(), delta_hidden1.end(), 0.0f);
        for(int j=0;j<hidden1_size;++j){
            float sum=0;
            for(int k=0;k<hidden2_size;++k)
                sum += delta_hidden2[k] * w1_2[k][j];
            delta_hidden1[j] = sum * hidden1[j] * (1.0f-hidden1[j]);
        }

        for(int j=0;j<hidden1_size;++j)
            for(int i=0;i<input_size;++i)
                grad_w0_1[j][i] = delta_hidden1[j] * input[i];

        for(int j=0;j<hidden2_size;++j)
            for(int i=0;i<hidden1_size;++i)
                grad_w1_2[j][i] = delta_hidden2[j] * hidden1[i];

        for(int j=0;j<output_size;++j)
            for(int i=0;i<hidden2_size;++i)
                grad_w2_3[j][i] = delta_output[j] * hidden2[i];
    }

    void applyGradients(){
        auto clip = [&](float &g){
            if(g>clip_th) g=clip_th;
            else if(g<-clip_th) g=-clip_th;
        };
        for(int j=0;j<hidden1_size;++j) for(int i=0;i<input_size;++i){
            clip(grad_w0_1[j][i]);
            vel_w0_1[j][i] = momentum*vel_w0_1[j][i] - eta*grad_w0_1[j][i];
            w0_1[j][i] += vel_w0_1[j][i];
        }
        for(int j=0;j<hidden2_size;++j) for(int i=0;i<hidden1_size;++i){
            clip(grad_w1_2[j][i]);
            vel_w1_2[j][i] = momentum*vel_w1_2[j][i] - eta*grad_w1_2[j][i];
            w1_2[j][i] += vel_w1_2[j][i];
        }
        for(int j=0;j<output_size;++j) for(int i=0;i<hidden2_size;++i){
            clip(grad_w2_3[j][i]);
            vel_w2_3[j][i] = momentum*vel_w2_3[j][i] - eta*grad_w2_3[j][i];
            w2_3[j][i] += vel_w2_3[j][i];
        }
        for(int i=0;i<hidden1_size;++i){
            clip(delta_hidden1[i]);
            vel_b1[i] = momentum*vel_b1[i] - eta*delta_hidden1[i];
            b1[i] += vel_b1[i];
        }
        for(int i=0;i<hidden2_size;++i){
            clip(delta_hidden2[i]);
            vel_b2[i] = momentum*vel_b2[i] - eta*delta_hidden2[i];
            b2[i] += vel_b2[i];
        }
        for(int i=0;i<output_size;++i){
            clip(delta_output[i]);
            vel_b3[i] = momentum*vel_b3[i] - eta*delta_output[i];
            b3[i] += vel_b3[i];
        }
    }

    void trainNetworkOneImageAtATime(
      const std::vector<std::vector<float>>& images,
      const std::vector<std::vector<float>>& labels,
      int numberOfImages, int epochs=1)
    {
        std::mt19937 rng(static_cast<uint32_t>(std::time(nullptr)));
        for(int e=0;e<epochs;++e){
            std::vector<int> idx(numberOfImages);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);

            correct = 0;
            for(int n : idx){
                forwardPass(images[n]);
                backProp(images[n], labels[n]);
                applyGradients();
                int pred = std::distance(output.begin(),
                             std::max_element(output.begin(), output.end()));
                int lab  = std::distance(labels[n].begin(),
                             std::max_element(labels[n].begin(), labels[n].end()));
                if(pred==lab) ++correct;
            }
            std::cout<<"Epoch "<<e+1
                     <<"  acc="<<100.0f*correct/numberOfImages<<"%\n";
        }
    }

    void trainNetworkInBatches(
      const std::vector<std::vector<float>>& images,
      const std::vector<std::vector<float>>& labels,
      int numberOfImages, int batchSize, int epochs=1)
    {
        std::mt19937 rng(static_cast<uint32_t>(std::time(nullptr)));
        for(int e=0;e<epochs;++e){
            std::vector<int> idx(numberOfImages);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);

            correct = 0;
            for(int start=0; start<numberOfImages; start+=batchSize){
                int end = std::min(start+batchSize, numberOfImages);
                // zero accumulators
                std::fill(grad_w0_1.begin(), grad_w0_1.end(), std::vector<float>(input_size,0.0f));
                std::fill(grad_w1_2.begin(), grad_w1_2.end(), std::vector<float>(hidden1_size,0.0f));
                std::fill(grad_w2_3.begin(), grad_w2_3.end(), std::vector<float>(hidden2_size,0.0f));
                std::fill(delta_hidden1.begin(), delta_hidden1.end(), 0.0f);
                std::fill(delta_hidden2.begin(), delta_hidden2.end(), 0.0f);
                std::fill(delta_output.begin(), delta_output.end(), 0.0f);

                int actualBatch = end - start;
                for(int bi=start; bi<end; ++bi){
                    int n = idx[bi];
                    forwardPass(images[n]);
                    backProp(images[n], labels[n]);
                    // accumulate grads
                    for(int j=0;j<hidden1_size;++j)
                      for(int i=0;i<input_size;++i)
                        grad_w0_1[j][i] += delta_hidden1[j] * images[n][i];
                    for(int j=0;j<hidden2_size;++j)
                      for(int i=0;i<hidden1_size;++i)
                        grad_w1_2[j][i] += delta_hidden2[j] * hidden1[i];
                    for(int j=0;j<output_size;++j)
                      for(int i=0;i<hidden2_size;++i)
                        grad_w2_3[j][i] += delta_output[j] * hidden2[i];

                    int pred = std::distance(output.begin(),
                                 std::max_element(output.begin(), output.end()));
                    int lab  = std::distance(labels[n].begin(),
                                 std::max_element(labels[n].begin(), labels[n].end()));
                    if(pred==lab) ++correct;
                }
                // average and apply
                float invB = 1.0f/actualBatch;
                for(int j=0;j<hidden1_size;++j)
                  for(int i=0;i<input_size;++i)
                    w0_1[j][i] -= eta * (grad_w0_1[j][i]*invB);
                for(int j=0;j<hidden2_size;++j)
                  for(int i=0;i<hidden1_size;++i)
                    w1_2[j][i] -= eta * (grad_w1_2[j][i]*invB);
                for(int j=0;j<output_size;++j)
                  for(int i=0;i<hidden2_size;++i)
                    w2_3[j][i] -= eta * (grad_w2_3[j][i]*invB);
            }
            std::cout<<"Epoch "<<e+1
                     <<"  acc="<<100.0f*correct/numberOfImages<<"%\n";
        }
    }

    void test(const std::string &png_path) {
        // 1) Load as grayscale
        int w,h,ch;
        unsigned char* data = stbi_load(png_path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            std::cerr<<"Failed to load "<<png_path<<"\n";
            return;
        }

        // 2) Compute center of mass
        double sumI=0, sumX=0, sumY=0;
        for(int y=0; y<h; ++y) for(int x=0; x<w; ++x){
            double v = 255 - data[y*w + x];
            sumI += v; sumX += x*v; sumY += y*v;
        }
        if(sumI<1e-6){ stbi_image_free(data); return; }
        double cx = sumX/sumI, cy = sumY/sumI;

        // 3) Compute moments for deskew
        double mu11=0, mu02=0;
        for(int y=0; y<h; ++y) for(int x=0; x<w; ++x){
            double v = 255 - data[y*w + x];
            double dx = x - cx, dy = y - cy;
            mu11 += dx*dy*v;
            mu02 += dy*dy*v;
        }
        double alpha = mu11 / (mu02 + 1e-8);

        // 4) Prepare target 28×28 image
        std::vector<float> img(28*28);

        // precompute scales
        double scaleX = double(w)/28.0, scaleY = double(h)/28.0;

        for(int ty=0; ty<28; ++ty){
            for(int tx=0; tx<28; ++tx){
                // map target → deskewed source
                double xd = (tx - 13.5)*scaleX + cx + alpha*( (ty - 13.5)*scaleY );
                double yd = (ty - 13.5)*scaleY + cy;
                int ix = std::clamp(int(std::round(xd)), 0, w-1);
                int iy = std::clamp(int(std::round(yd)), 0, h-1);
                unsigned char px = data[iy*w + ix];
                // binarize & normalize: fill small gaps
                float v = (255 - px)/255.0f;
                img[ty*28 + tx] = (v > 0.1f ? 0.5f : -0.5f);
            }
        }
        stbi_image_free(data);

        // 5) Morphological closing (3×3 dilate then erode)
        auto dilated = img;
        for(int y=1; y<27; ++y) for(int x=1; x<27; ++x){
            float m = -1;
            for(int dy=-1; dy<=1; ++dy)
              for(int dx=-1; dx<=1; ++dx)
                m = std::max(m, img[(y+dy)*28 + (x+dx)]);
            dilated[y*28 + x] = m;
        }
        auto closed = dilated;
        for(int y=1; y<27; ++y) for(int x=1; x<27; ++x){
            float m = 1;
            for(int dy=-1; dy<=1; ++dy)
              for(int dx=-1; dx<=1; ++dx)
                m = std::min(m, dilated[(y+dy)*28 + (x+dx)]);
            closed[y*28 + x] = m;
        }
        img.swap(closed);

        // 6) Forward pass
        forwardPass(img);

        // 7) Pick max
        int pred = 0;
        float best = output[0];
        for(int i=1; i<output_size; ++i){
            if(output[i]>best){ best=output[i]; pred=i; }
        }

        std::cout<<"Test \""<<png_path<<"\"predicted: "<<pred<<"\n";
    }
    
};

int main(){
    std::cout << "Starting...\n" << std::flush;
    srand(time(0));
    std::string base = "C:\\Users\\samfr\\OneDrive\\Desktop\\Digit NN\\data\\";
    auto data = loadMNIST(base+"t10k-images.idx3-ubyte", base+"t10k-labels.idx1-ubyte");
    auto x = normaliseImages(data.images);
    auto y = oneHotLabels(data.labels);

    NeuralNetwork net;
    net.loadParameters();
    net.trainNetworkOneImageAtATime(x, y, data.num_images, 3);
    net.trainNetworkInBatches(x, y, data.num_images, 32, 2);
    net.saveParameters();
    net.test("C:\\Users\\samfr\\OneDrive\\Desktop\\Digit NN\\data\\Test_Digit=4.png");
    net.test("C:\\Users\\samfr\\OneDrive\\Desktop\\Digit NN\\data\\Test_Digit=8.png");
    net.test("C:\\Users\\samfr\\OneDrive\\Desktop\\Digit NN\\data\\Test_Digit=0.png");
    return 0;
}
