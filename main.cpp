#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include<opencv2/core/eigen.hpp>
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
        cv::rectangle(*result_rgb,cv::Rect(i*width, 0, width, height),cv::Scalar::all(255),1);
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

void make_patch_with_warp(int halfpatch_size, Eigen::Vector2d uvs, cv::Mat mats, uint8_t* patch_ptr_b, Eigen::Matrix3d Wts){
    const int strideB = mats.step.p[0];
    Eigen::Matrix3d Winv=Wts.inverse();
    Eigen::Vector3d uvt= Wts * Eigen::Vector3d(uvs.x(), uvs.y(), 1.0);
    uvt/=uvt.z();
    for (int y=-halfpatch_size; y<halfpatch_size; ++y)
    {
        for (int x=-halfpatch_size; x<halfpatch_size; ++x,++patch_ptr_b)
        {
            const Eigen::Vector3d px_patch(x + uvt.x(), y + uvt.y(), 1.0);
            Eigen::Vector3d wpx=Winv*px_patch;
            wpx/=wpx.z();
            //std::cout<<wpx.transpose()<<std::endl;
            const Eigen::Vector2d px(wpx.topLeftCorner<2,1>());
            const int xi = std::floor(px[0]);
            const int yi = std::floor(px[1]);
            if (xi<0 || yi<0 || xi+1 >= mats.cols || yi + 1 >= mats.rows)
                continue;
            else
            {
                const float subpix_x = px[0]-xi;
                const float subpix_y = px[1]-yi;
                const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
                const float w01 = (1.0f-subpix_x)*subpix_y;
                const float w10 = subpix_x*(1.0f-subpix_y);
                const float w11 = 1.0f - w00 - w01 - w10;
                const uint8_t* const ptr = mats.data + yi * strideB + xi;
                *patch_ptr_b = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[strideB] + w10*ptr[1] + w11*ptr[strideB+1]);
            }
        }
    }
}

inline void halfSample(const cv::Mat& in, cv::Mat& out)
{
    assert( in.rows/2==out.rows && in.cols/2==out.cols);
    assert( in.type()==CV_8U && out.type()==CV_8U);

    const int in_stride = in.step.p[0];
    const int out_stride = out.step.p[0];
    uint8_t* top = (uint8_t*) in.data;
    uint8_t* bottom = top + in_stride;
    uint8_t* end = top + in_stride*in.rows;
    uint8_t* p = (uint8_t*) out.data;
    for (int y=0; y < out.rows && bottom < end; y++, top += in_stride*2, bottom += in_stride*2, p += out_stride)
    {
        for (int x=0; x < out.cols; x++)
        {
            p[x] = static_cast<uint8_t>( (uint16_t (top[x*2]) + top[x*2+1] + bottom[x*2] + bottom[x*2+1])/4 );
            if(p[x]>60){
                p[x]=255;
            }
        }
    }
}
using ImgPyr=std::vector<cv::Mat>;
inline void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize(n_levels);
    pyr[0] = img_level_0;
    for(int i=1; i<n_levels; ++i)
    {
        pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
        halfSample(pyr[i-1], pyr[i]);
    }
}

inline int getBestSearchLevel(
        const Eigen::Matrix2d& A_cur_ref,
        const int max_level)
{
    // Compute patch level in other image
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while(D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

int main() {
    std::vector<Eigen::Vector3d> circle;//in A frame,ref
    float r=5.0;
    for (int i = 0; i < 360; ++i) {
        Eigen::Vector3d p;
        p.z()=10;
        p.x()=r*cos(float(i)*M_PI/180.0);
        p.y()=r*sin(float(i)*M_PI/180.0);
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
//            if(i==0 || i==100 || i==200){
//                cv::circle(matA,cv::Point(uv.x(),uv.y()),4,cv::Scalar(200),3);
//            }
            matA.at<uchar>(uv.y(),uv.x())=255;
            matA.at<uchar>(uv.y()+1,uv.x()+1)=255;
            matA.at<uchar>(uv.y()-1,uv.x()-1)=255;
            matA.at<uchar>(uv.y()-1,uv.x()+1)=255;
            matA.at<uchar>(uv.y()+1,uv.x()-1)=255;
            matA.at<uchar>(uv.y()+1,uv.x())=255;
            matA.at<uchar>(uv.y(),uv.x()+1)=255;
            matA.at<uchar>(uv.y(),uv.x()-1)=255;
            matA.at<uchar>(uv.y()-1,uv.x())=255;
        }
    }

    Eigen::Vector3d t(10,0,0);
    Eigen::AngleAxisd ry=Eigen::AngleAxisd(-M_PI/4.0,Eigen::Vector3d::UnitY());

//    Eigen::Vector3d t(0,0,0);
//    Eigen::AngleAxisd ry=Eigen::AngleAxisd(0.0,Eigen::Vector3d::UnitY());

    Eigen::Matrix4d T_BA;
    T_BA.setIdentity();
    T_BA.topLeftCorner<3,3>()=ry.toRotationMatrix().transpose();
    T_BA.topRightCorner<3,1>()=-ry.toRotationMatrix().transpose()*t;
    cv::Mat matB=cv::Mat::zeros(size,size,CV_8UC1);//cur
    for (int i = 0; i < circle.size(); ++i) {
        Eigen::Vector3d p=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
        p/=p.z();
        Eigen::Vector2d uv=(K*p).topLeftCorner<2,1>();
        if(uv.x()>=0 && uv.x()<size && uv.y()>=0 && uv.y()<size ){
            matB.at<uchar>(uv.y(),uv.x())=255;
//            if(i==0 || i==100 || i==200){
//                cv::circle(matB,cv::Point(uv.x(),uv.y()),4,cv::Scalar(200),3);
//            }
            matB.at<uchar>(uv.y()+1,uv.x()+1)=255;
            matB.at<uchar>(uv.y()-1,uv.x()-1)=255;
            matB.at<uchar>(uv.y()-1,uv.x()+1)=255;
            matB.at<uchar>(uv.y()+1,uv.x()-1)=255;
            matB.at<uchar>(uv.y()+1,uv.x())=255;
            matB.at<uchar>(uv.y(),uv.x()+1)=255;
            matB.at<uchar>(uv.y(),uv.x()-1)=255;
            matB.at<uchar>(uv.y()-1,uv.x())=255;
        }
    }

    //warp affine:

    //Eigen::Vector3d origin(0,0,10);
    Eigen::Vector3d p0=circle[40];//in A
    double dis=p0.z();
    Eigen::Vector2d uv0=(K*(p0/p0.z())).topLeftCorner<2,1>();

    Eigen::Vector3d p1=circle[80];//in A
    Eigen::Vector2d uv1=(K*(p1/p1.z())).topLeftCorner<2,1>();

    Eigen::Vector3d p2=circle[120];//in A
    Eigen::Vector2d uv2=(K*(p2/p2.z())).topLeftCorner<2,1>();

    Eigen::Vector3d p3=circle[160];//in A
    Eigen::Vector2d uv3=(K*(p3/p3.z())).topLeftCorner<2,1>();

    //std::cout<<"origin uv "<<(K*(origin/origin.z())).topLeftCorner<2,1>().transpose()<<std::endl;
    Eigen::Vector3d pdu=Kinv*Eigen::Vector3d(uv0.x()+10.0,uv0.y(),1.0)*dis;//in A, ref
    Eigen::Vector3d pdv=Kinv*Eigen::Vector3d(uv0.x(),uv0.y()+10.0,1.0)*dis;
    //std::cout<<" error "<<Kinv*Eigen::Vector3d(uv0.x(),uv0.y(),1.0)*dis-p0<<std::endl;

    Eigen::Vector3d pdu_B=ry.toRotationMatrix().transpose()*pdu-ry.toRotationMatrix().transpose()*t;//in B,cur
    pdu_B/=pdu_B.z();
    Eigen::Vector3d pdv_B=ry.toRotationMatrix().transpose()*pdv-ry.toRotationMatrix().transpose()*t;
    pdv_B/=pdv_B.z();
    Eigen::Vector3d p_B=ry.toRotationMatrix().transpose()*p0-ry.toRotationMatrix().transpose()*t;
    p_B/=p_B.z();
    //Eigen::Vector3d origin_b=ry.toRotationMatrix().transpose()*origin-ry.toRotationMatrix().transpose()*t;
    //std::cout<<"origin uv b  "<<(K*(origin_b/origin_b.z())).topLeftCorner<2,1>().transpose()<<std::endl;


    auto fdu=(K*pdu_B).topLeftCorner<2,1>();
    auto fdv=(K*pdv_B).topLeftCorner<2,1>();
    auto f=(K*p_B).topLeftCorner<2,1>();
    Eigen::Matrix2d W;//W_BA,aka A_cur_ref
    W.setZero();
    W.col(0) = (fdu - f) / 10.0;
    W.col(1) = (fdv - f) / 10.0;

    std::cout<<W<<std::endl;
    std::cout<<"W "<<W.determinant()<<std::endl;


    //warp perspective by match
    Eigen::Vector3d p0_B=ry.toRotationMatrix().transpose()*p0-ry.toRotationMatrix().transpose()*t;//in B,cur
    p0_B/=p0_B.z();
    Eigen::Vector2d uv0_b=(K*p0_B).topLeftCorner<2,1>();

    Eigen::Vector3d p1_B=ry.toRotationMatrix().transpose()*p1-ry.toRotationMatrix().transpose()*t;//in B,cur
    p1_B/=p1_B.z();
    Eigen::Vector2d uv1_b=(K*p1_B).topLeftCorner<2,1>();

    Eigen::Vector3d p2_B=ry.toRotationMatrix().transpose()*p2-ry.toRotationMatrix().transpose()*t;//in B,cur
    p2_B/=p2_B.z();
    Eigen::Vector2d uv2_b=(K*p2_B).topLeftCorner<2,1>();

    Eigen::Vector3d p3_B=ry.toRotationMatrix().transpose()*p3-ry.toRotationMatrix().transpose()*t;//in B,cur
    p3_B/=p3_B.z();
    Eigen::Vector2d uv3_b=(K*p3_B).topLeftCorner<2,1>();


    cv::Point2f srcTri[4];
    srcTri[0] = cv::Point2f( uv0.x(),uv0.y() );
    srcTri[1] = cv::Point2f( uv1.x(),uv1.y());
    srcTri[2] = cv::Point2f( uv2.x(),uv2.y() );
    srcTri[3] = cv::Point2f( uv3.x(),uv3.y() );
    cv::Point2f dstTri[4];
    dstTri[0] = cv::Point2f( uv0_b.x(),uv0_b.y());
    dstTri[1] = cv::Point2f( uv1_b.x(),uv1_b.y() );
    dstTri[2] = cv::Point2f( uv2_b.x(),uv2_b.y() );
    dstTri[3] = cv::Point2f( uv3_b.x(),uv3_b.y() );

    cv::Mat warp_BA = cv::getPerspectiveTransform(  srcTri,dstTri );//A_ba
    std::cout<<" opencv "<<std::endl;
    std::cout<<warp_BA<<std::endl;




    //warp perspective by pseduo-match
    Eigen::Vector3d uv0px(uv0.x()+10,uv0.y(),1.0);
    Eigen::Vector3d uv0mx(uv0.x()-10,uv0.y(),1.0);
    Eigen::Vector3d uv0py(uv0.x(),uv0.y()+10,1.0);
    Eigen::Vector3d uv0my(uv0.x(),uv0.y()-10,1.0);
    Eigen::Vector3d puv0px=Kinv*uv0px*dis;//in A, ref
    Eigen::Vector3d puv0mx=Kinv*uv0mx*dis;
    Eigen::Vector3d puv0my=Kinv*uv0my*dis;
    Eigen::Vector3d puv0py=Kinv*uv0py*dis;

    Eigen::Vector3d puv0px_B=ry.toRotationMatrix().transpose()*puv0px-ry.toRotationMatrix().transpose()*t;//in B,cur
    puv0px_B/=puv0px_B.z();
    Eigen::Vector2d uv0px_b=(K*puv0px_B).topLeftCorner<2,1>();

    Eigen::Vector3d puv0py_B=ry.toRotationMatrix().transpose()*puv0py-ry.toRotationMatrix().transpose()*t;//in B,cur
    puv0py_B/=puv0py_B.z();
    Eigen::Vector2d uv0py_b=(K*puv0py_B).topLeftCorner<2,1>();

    Eigen::Vector3d puv0mx_B=ry.toRotationMatrix().transpose()*puv0mx-ry.toRotationMatrix().transpose()*t;//in B,cur
    puv0mx_B/=puv0mx_B.z();
    Eigen::Vector2d uv0mx_b=(K*puv0mx_B).topLeftCorner<2,1>();

    Eigen::Vector3d puv0my_B=ry.toRotationMatrix().transpose()*puv0my-ry.toRotationMatrix().transpose()*t;//in B,cur
    puv0my_B/=puv0my_B.z();
    Eigen::Vector2d uv0my_b=(K*puv0my_B).topLeftCorner<2,1>();

    {
        double s=2.0;
        cv::Point2f srcTri[5];
        srcTri[0] = cv::Point2f( uv0px.x()/s,uv0px.y()/s );
        srcTri[1] = cv::Point2f( uv0mx.x()/s,uv0mx.y()/s);
        srcTri[2] = cv::Point2f( uv0py.x()/s,uv0py.y()/s );
        srcTri[3] = cv::Point2f( uv0my.x()/s,uv0my.y()/s );
        //srcTri[4] = cv::Point2f( uv0.x(),uv0.y() );
        cv::Point2f dstTri[5];
        dstTri[0] = cv::Point2f( uv0px_b.x(),uv0px_b.y());
        dstTri[1] = cv::Point2f( uv0mx_b.x(),uv0mx_b.y() );
        dstTri[2] = cv::Point2f( uv0py_b.x(),uv0py_b.y() );
        dstTri[3] = cv::Point2f( uv0my_b.x(),uv0my_b.y() );
        //dstTri[4] = cv::Point2f( uv0_b.x(),uv0_b.y() );

        double n=0;
        n+=cv::norm(srcTri[0]-srcTri[2])/cv::norm(dstTri[0]-dstTri[2]);
        n+=cv::norm(srcTri[1]-srcTri[3])/cv::norm(dstTri[1]-dstTri[3]);
        std::cout<<"scale "<<n/2.0<<std::endl;

        cv::Mat warp_BA = cv::getPerspectiveTransform(  srcTri,dstTri );//A_ba
        std::cout<<" opencv 2 "<<std::endl;
        std::cout<<warp_BA<<std::endl;
    }




    Eigen::Matrix<double,3,3> Wba;
    Wba.setZero();
    cv::cv2eigen(warp_BA,Wba);
    std::cout<<"Wba "<<Wba.topLeftCorner<2,2>().determinant()<<std::endl;

    cv::Mat warpedB = cv::Mat::zeros( matA.rows, matA.cols, matA.type() );
    for (int i = 0; i < matA.cols; ++i) {
        for (int j = 0; j < matA.rows; ++j) {
            Eigen::Vector3d uv=(Wba*Eigen::Vector3d(i,j,1.0));
            uv/=uv.z();
            if(uv.x()<0 || uv.y()<0 || uv.x()>matA.cols-1 || uv.y()>matA.rows-1){
                continue;
            }
            warpedB.at<uchar>(int(uv.y()),int(uv.x()))=matA.at<uchar>(j,i);
        }
    }

    //cv::warpPerspective( matA, warpedB, warp_BA, warpedB.size() );
    cv::imwrite("warpedB.jpg",warpedB);
    cv::imwrite("A.jpg",matA);
    cv::imwrite("B.jpg",matB);



    ImgPyr imgAPyr,imgBPyr;
    createImgPyramid(matA,4,imgAPyr);
    createImgPyramid(matB,4,imgBPyr);

    const int halfpatch_size=15;
    const int patch_size=halfpatch_size*2;


    for (int level = 0; level < 4; ++level) {
        double scale=1.36;
        std::cout<<"current level "<<level<<std::endl;
        scale/=(1<<level);
        int search_level= std::log2(1.0/scale);
        if(search_level>3)
            search_level=3;
        std::cout<<"best search level "<<search_level<<std::endl;

        cv::Point2f srcTri[4];
        srcTri[0] = cv::Point2f( uv0.x()/(1<<level),uv0.y()/(1<<level) );
        srcTri[1] = cv::Point2f( uv1.x()/(1<<level),uv1.y()/(1<<level));
        srcTri[2] = cv::Point2f( uv2.x()/(1<<level),uv2.y()/(1<<level) );
        srcTri[3] = cv::Point2f( uv3.x()/(1<<level),uv3.y()/(1<<level) );
        cv::Point2f dstTri[4];
        dstTri[0] = cv::Point2f( uv0_b.x()/(1<<search_level),uv0_b.y()/(1<<search_level));
        dstTri[1] = cv::Point2f( uv1_b.x()/(1<<search_level),uv1_b.y()/(1<<search_level) );
        dstTri[2] = cv::Point2f( uv2_b.x()/(1<<search_level),uv2_b.y()/(1<<search_level) );
        dstTri[3] = cv::Point2f( uv3_b.x()/(1<<search_level),uv3_b.y()/(1<<search_level) );

        cv::Mat warp_BA = cv::getPerspectiveTransform(  srcTri,dstTri );//A_ba
        std::cout<<" opencv "<<std::endl;
        std::cout<<warp_BA<<std::endl;


        Eigen::Matrix<double,3,3> Wlevel;
        Wlevel.setZero();
        cv::cv2eigen(warp_BA,Wlevel);




        std::vector<cv::Mat> ref_patches,cur_patches;
        for (int i = 0; i < circle.size(); i++) {
            Eigen::Vector3d pa=circle[i];
            pa/=pa.z();
            Eigen::Vector2d uva=(K*pa).topLeftCorner<2,1>()/(1<<level);


            uint8_t A_patch_ptr[patch_size*patch_size]  __attribute__ ((aligned (16)));
            uint8_t* patch_ptr_a = A_patch_ptr;
            make_patch_with_warp(halfpatch_size,uva,imgAPyr[level],patch_ptr_a,Wlevel);//ref



            Eigen::Vector3d pb=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
            pb/=pb.z();
            Eigen::Vector2d uvb=(K*pb).topLeftCorner<2,1>()/(1<<search_level);

            uint8_t B_patch_ptr[patch_size*patch_size]  __attribute__ ((aligned (16)));
            uint8_t* patch_ptr_b = B_patch_ptr;

            make_patch(halfpatch_size,uvb,imgBPyr[search_level],patch_ptr_b);




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
        cv::imwrite("level"+std::to_string(level)+"res.jpg",full_imag);
        cv::imwrite("level"+std::to_string(level)+"A.jpg",imgAPyr[level]);
        cv::imwrite("level"+std::to_string(level)+"B.jpg",imgBPyr[level]);
    }


    return 0;
}
