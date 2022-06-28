#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
inline void patchToMat(
        const uint8_t* const patch_data,
        const size_t patch_width,
        cv::Mat* img)
{
    *img = cv::Mat(patch_width, patch_width, CV_8UC1);
    std::memcpy(img->data, patch_data, patch_width*patch_width);
}

inline void concatenatePatches(
        std::vector<cv::Mat> patches,
        cv::Mat* result_rgb)
{
    size_t n = patches.size();
    const int width = patches.at(0).cols;
    const int height = patches.at(0).rows;

    *result_rgb = cv::Mat(height, n*width, CV_8UC1);
    for(size_t i = 0; i < n; ++i)
    {
        cv::Mat roi(*result_rgb, cv::Rect(i*width, 0, width, height));
        cv::Mat patch_rgb(patches[i].size(), CV_8UC1);
        //cv::cvtColor(patches[i], patch_rgb, cv::COLOR_GRAY2RGB);
        patch_rgb=patches[i].clone();
        patch_rgb.copyTo(roi);
    }
}

inline void make_patch(int halfpatch_size,Eigen::Vector2d uva,cv::Mat matA,uint8_t* patch_ptr_a){
    const int strideA = matA.step.p[0];
    for (int y=-halfpatch_size; y<halfpatch_size; ++y)
    {
        for (int x=-halfpatch_size; x<halfpatch_size; ++x,++patch_ptr_a)
        {
            const Eigen::Vector2d px_patch(x, y);
            const Eigen::Vector2d px(px_patch + uva);
            const int xi = std::floor(px[0]);
            const int yi = std::floor(px[1]);
            if (xi<0 || yi<0 || xi+1>=matA.cols || yi+1>=matA.rows)
                continue;
            else
            {
                const float subpix_x = px[0]-xi;
                const float subpix_y = px[1]-yi;
                const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
                const float w01 = (1.0f-subpix_x)*subpix_y;
                const float w10 = subpix_x*(1.0f-subpix_y);
                const float w11 = 1.0f - w00 - w01 - w10;
                const uint8_t* const ptr = matA.data + yi*strideA + xi;
                *patch_ptr_a = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[strideA] + w10*ptr[1] + w11*ptr[strideA+1]);
            }
        }
    }
}

void make_patch_with_warp(int halfpatch_size,Eigen::Vector2d uvb,cv::Mat matB,uint8_t* patch_ptr_b,Eigen::Matrix2d A){
    const int strideB = matB.step.p[0];
    for (int y=-halfpatch_size; y<halfpatch_size; ++y)
    {
        for (int x=-halfpatch_size; x<halfpatch_size; ++x,++patch_ptr_b)
        {
            const Eigen::Vector2d px_patch(x, y);
            const Eigen::Vector2d px(A*px_patch + uvb);
            const int xi = std::floor(px[0]);
            const int yi = std::floor(px[1]);
            if (xi<0 || yi<0 || xi+1>=matB.cols || yi+1>=matB.rows)
                continue;
            else
            {
                const float subpix_x = px[0]-xi;
                const float subpix_y = px[1]-yi;
                const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
                const float w01 = (1.0f-subpix_x)*subpix_y;
                const float w10 = subpix_x*(1.0f-subpix_y);
                const float w11 = 1.0f - w00 - w01 - w10;
                const uint8_t* const ptr = matB.data + yi*strideB + xi;
                *patch_ptr_b = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[strideB] + w10*ptr[1] + w11*ptr[strideB+1]);
            }
        }
    }
}
int main() {
    std::vector<Eigen::Vector3d> circle;//in A frame
    float r=3.0;
    for (int i = 0; i < 180; ++i) {
        Eigen::Vector3d p;
        p.z()=10;
        p.x()=r*cos(2.0*float(i)*M_PI/180.0);
        p.y()=r*sin(2.0*float(i)*M_PI/180.0);
        circle.push_back(p);
    }
    Eigen::Matrix3d K,Kinv;
    K<<400.0,0,500.0,
        0,400.0,500.0,
        0,0,1.0;
    Kinv<<1.0,0,-500,
    0,1,-500,
    0,0,400.0;
    Kinv*=(1.0/400.0);
    int size=1000;
    cv::Mat matA=cv::Mat::zeros(size,size,CV_8UC1);
    for (int i = 0; i < circle.size(); ++i) {
        Eigen::Vector3d p=circle[i];
        p/=p.z();
        Eigen::Vector2d uv=(K*p).topLeftCorner<2,1>();
        if(uv.x()>=0 && uv.x()<size && uv.y()>=0 && uv.y()<size ){
            matA.at<uchar>(uv.y(),uv.x())=255;
        }
    }

    Eigen::Vector3d t(20,0,0);
    Eigen::AngleAxisd ry=Eigen::AngleAxisd(-M_PI/4.0,Eigen::Vector3d::UnitY());

//    Eigen::Vector3d t(0,0,0);
//    Eigen::AngleAxisd ry=Eigen::AngleAxisd(0.0,Eigen::Vector3d::UnitY());

    Eigen::Matrix4d T_BA;
    T_BA.setIdentity();
    T_BA.topLeftCorner<3,3>()=ry.toRotationMatrix().transpose();
    T_BA.topRightCorner<3,1>()=-ry.toRotationMatrix().transpose()*t;
    cv::Mat matB=cv::Mat::zeros(size,size,CV_8UC1);;
    for (int i = 0; i < circle.size(); ++i) {
        Eigen::Vector3d p=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
        p/=p.z();
        Eigen::Vector2d uv=(K*p).topLeftCorner<2,1>();
        if(uv.x()>=0 && uv.x()<size && uv.y()>=0 && uv.y()<size ){
            matB.at<uchar>(uv.y(),uv.x())=255;
        }
    }

//warp B


    Eigen::Vector3d p0=circle[1];//in A
    double dis=p0.z();
    Eigen::Vector2d uv0=(K*(p0/p0.z())).topLeftCorner<2,1>();
    Eigen::Vector3d pdu=Kinv*Eigen::Vector3d(uv0.x()+10.0,uv0.y(),1.0)*dis;//in A
    Eigen::Vector3d pdv=Kinv*Eigen::Vector3d(uv0.x(),uv0.y()+10.0,1.0)*dis;
    //std::cout<<" error "<<Kinv*Eigen::Vector3d(uv0.x(),uv0.y(),1.0)*dis-p0<<std::endl;

    Eigen::Vector3d pdu_B=ry.toRotationMatrix().transpose()*pdu-ry.toRotationMatrix().transpose()*t;//in B
    pdu_B/=pdu_B.z();
    Eigen::Vector3d pdv_B=ry.toRotationMatrix().transpose()*pdv-ry.toRotationMatrix().transpose()*t;
    pdv_B/=pdv_B.z();
    Eigen::Vector3d p_B=ry.toRotationMatrix().transpose()*p0-ry.toRotationMatrix().transpose()*t;
    p_B/=p_B.z();


    auto fdu=(K*pdu_B).topLeftCorner<2,1>();
    auto fdv=(K*pdv_B).topLeftCorner<2,1>();
    auto f=(K*p_B).topLeftCorner<2,1>();
    Eigen::Matrix2d W;//W_BA
    W.setZero();
    W.col(0) = (fdu - f) / 10.0;
    W.col(1) = (fdv - f) / 10.0;

    W(1,0)=0.0;

    std::cout<<W.determinant()<<std::endl;

    const int halfpatch_size=15;
    const int patch_size=halfpatch_size*2;
    std::vector<cv::Mat> ref_patches,cur_patches;
    for (int i = 0; i < circle.size(); i++) {
        Eigen::Vector3d pa=circle[i];
        pa/=pa.z();
        Eigen::Vector2d uva=(K*pa).topLeftCorner<2,1>();


        uint8_t A_patch_ptr[patch_size*patch_size]  __attribute__ ((aligned (16)));
        uint8_t* patch_ptr_a = A_patch_ptr;
        make_patch(halfpatch_size,uva,matA,patch_ptr_a);



        Eigen::Vector3d pb=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
        pb/=pb.z();
        Eigen::Vector2d uvb=(K*pb).topLeftCorner<2,1>();

        uint8_t B_patch_ptr[patch_size*patch_size]  __attribute__ ((aligned (16)));
        uint8_t* patch_ptr_b = B_patch_ptr;

        make_patch_with_warp(halfpatch_size,uvb,matB,patch_ptr_b,W);




        cv::Mat ref_img_mat;
        patchToMat(patch_ptr_a,patch_size,&ref_img_mat);
        ref_patches.push_back(ref_img_mat);

        cv::Mat cur_img_mat;
        patchToMat(patch_ptr_b,patch_size,&cur_img_mat);
        cur_patches.push_back(cur_img_mat);

    }

    cv::Mat debug_ref_img,debug_cur_img,full_imag;
    concatenatePatches(ref_patches,&debug_ref_img);
    concatenatePatches(cur_patches,&debug_cur_img);
    cv::vconcat(debug_ref_img,debug_cur_img,full_imag);

//    cv::namedWindow("circle",cv::WINDOW_FULLSCREEN);
//    cv::imshow("circle",full_imag);
//    cv::waitKey(-1);
   cv::imwrite("res.jpg",full_imag);
    cv::imwrite("A.jpg",matA);
    cv::imwrite("B.jpg",matB);

    return 0;
}
