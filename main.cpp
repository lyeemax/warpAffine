#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include<opencv2/core/eigen.hpp>
#include "sophus/se3.hpp"
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

inline bool computeCurrentFeaturePatch(const Eigen::Vector2d &feature, const int &patch_size, const cv::Mat & img, uint8_t* cur_patch_array, float *jacobian_dx, float *jacobian_dy){
    const int rows_minus_two = img.rows - 2;
    const int cols_minus_two = img.cols - 2;
    const int stride = img.step.p[0]; // must be real stride
    const int border = 1;
    const int patch_size_wb = patch_size + 2*border; //patch size with border
    const int patch_area_wb = patch_size_wb*patch_size_wb;
    const int halfpatch_size_wb=patch_size_wb/2.0;

    const int strideA = img.step.p[0];
    // interpolate patch + border (filled in row major format)
    uint8_t interp_patch_array [patch_area_wb];
    uint8_t* p_interp_patch_array =interp_patch_array;
    for (int y=-halfpatch_size_wb; y<halfpatch_size_wb; ++y)
    {
        for (int x=-halfpatch_size_wb; x<halfpatch_size_wb; ++x,++p_interp_patch_array)
        {
            const Eigen::Vector2d px_patch(x, y);
            const Eigen::Vector2d px=px_patch + feature;
            const int xi = std::floor(px[0]);
            const int yi = std::floor(px[1]);
            if (xi<0 || yi<0 || xi+1>=img.cols || yi+1>=img.rows)
                return false;
            else
            {
                const float subpix_x = px[0]-xi;
                const float subpix_y = px[1]-yi;
                const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
                const float w01 = (1.0f-subpix_x)*subpix_y;
                const float w10 = subpix_x*(1.0f-subpix_y);
                const float w11 = 1.0f - w00 - w01 - w10;
                const uint8_t* const ptr = img.data + yi*strideA + xi;
                *p_interp_patch_array = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[strideA] + w10*ptr[1] + w11*ptr[strideA+1]);
            }
        }
    }

    // fill ref_patch_cache and jacobian_cache
    size_t pixel_counter = 0;
    for(int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x, ++pixel_counter) {
            int offset_center = (x + border) + patch_size_wb *(y+border);
            cur_patch_array[pixel_counter] =
                    interp_patch_array[offset_center];
            const double dx = static_cast<double>(0.5f*(interp_patch_array[offset_center + 1]
                                    -interp_patch_array[offset_center - 1]));
            const double dy = static_cast<double>(0.5f*(interp_patch_array[offset_center + patch_size_wb]
                                    -interp_patch_array[offset_center - patch_size_wb]));
            if(jacobian_dx && jacobian_dy){
                jacobian_dx[pixel_counter]=dx;
                jacobian_dy[pixel_counter]=dy;
            }
        }
    }
    return true;
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

bool make_patch_with_warp(int halfpatch_size, Eigen::Vector2d uvs, cv::Mat mats, uint8_t* patch_ptr_b, Eigen::Matrix3d Wts){
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
                return false;
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
    return true;
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

inline double get_huber_loss_scale( double reprojection_error, double outlier_threshold)
{
    double scale = 1.0;
    if ( reprojection_error / outlier_threshold < 1.0 )
    {
        scale = 1.0;
    }
    else
    {
        scale = ( 2 * sqrt( reprojection_error ) / sqrt( outlier_threshold ) - 1.0 ) / reprojection_error;
    }
    return scale;
}

double linearize(const std::vector<Eigen::Vector3d> &circle,const int &level,const Sophus::SE3d &TBA,const Eigen::Matrix3d &Wlevel,const Eigen::Matrix3d &K,const int &search_level,
                 const int& patch_size, const int &halfpatch_size,const std::vector<cv::Mat> &imgAPyr,const std::vector<cv::Mat> &imgBPyr,const bool & calJ,
                 int& idx,std::vector<std::vector<Eigen::MatrixXd>> &Js,std::vector<std::vector<float>> &Res,std::vector<int>& vidx,bool init) {
    double sum_res = 0;
    int obs_size=circle.size();
    if(!init){
        obs_size=vidx.size();
    }
    for (int i = 0; i < obs_size; i++) {
        int pid=i;
        if(!init){
            pid=vidx[i];
        }
        Eigen::Vector3d pa = circle[pid];
        pa /= pa.z();
        Eigen::Vector2d uva = (K * pa).topLeftCorner<2, 1>() / (1 << level);


        uint8_t A_patch_ptr[patch_size * patch_size]  __attribute__ ((aligned (16)));
        uint8_t *patch_ptr_a = A_patch_ptr;
        bool res=make_patch_with_warp(halfpatch_size, uva, imgAPyr[level], patch_ptr_a, Wlevel);//ref
        if(!init){
            res=true;
        }
        if(!res)
            continue;

        //Eigen::Vector3d pb=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
        Eigen::Vector3d pb = TBA * circle[pid];
        Eigen::Vector2d uvb = (K * pb / pb.z()).topLeftCorner<2, 1>() / (1 << search_level);

        uint8_t B_patch_ptr[patch_size * patch_size]  __attribute__ ((aligned (16)));
        uint8_t *patch_ptr_b = B_patch_ptr;
        float __attribute__((__aligned__(16))) cur_patch_dx[patch_size * patch_size];
        float __attribute__((__aligned__(16))) cur_patch_dy[patch_size * patch_size];
        bool res1=computeCurrentFeaturePatch(uvb, patch_size, imgBPyr[search_level], patch_ptr_b, cur_patch_dx, cur_patch_dy);
        if(!init){
            res=true;
        }
        if(!res1)
            continue;
        Eigen::Matrix<double, 2, 3> mat_pre;
        Eigen::Matrix<double, 3, 6> mat_d_f_d_T;
        if (calJ) {
            mat_pre << K(0, 0), 0, -K(0, 0) * pb(0) / pb(2),
                    0, K(1, 1), -K(1, 1) * pb(1) / pb(2);
            mat_pre /= pb.z();
            mat_d_f_d_T.setZero();
            mat_d_f_d_T.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
            mat_d_f_d_T.topRightCorner<3, 3>() = -TBA.rotationMatrix() * Sophus::SO3d::hat(circle[pid]);

            Js[idx].resize(patch_size * patch_size);
            Res[idx].resize(patch_size * patch_size);
        }

        for (int j = 0; j < patch_size * patch_size; ++j) {
            double intensity_error = static_cast<double>(patch_ptr_b[j] - patch_ptr_a[j]);
            intensity_error *= get_huber_loss_scale(intensity_error, 10);
            sum_res += intensity_error * intensity_error;
            if (calJ) {
                Eigen::Matrix<double, 1, 2> mat_photometric;
                Eigen::Matrix<double, 1, 3> mat_d_pho_d_img;
                mat_photometric.setZero();
                mat_photometric[0] = cur_patch_dx[j];
                mat_photometric[1] = cur_patch_dy[j];
                Eigen::Matrix<double, 1, 6> J=mat_photometric * mat_pre * mat_d_f_d_T / (1 << search_level);
                Js[idx][j] = J;
                Res[idx][j] = intensity_error;
            }
        }
        if(init){
            vidx.emplace_back(i);
        }
        idx++;
    }
    return sum_res;
}
inline void createPatchFromPatchWithBorder(
        const uint8_t* const patch_with_border,
        const int patch_size,
        uint8_t* patch)
{
    uint8_t* patch_ptr = patch;
    for(int y=1; y<patch_size+1; ++y, patch_ptr += patch_size)
    {
        const uint8_t* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
        for(int x=0; x<patch_size; ++x)
            patch_ptr[x] = ref_patch_border_ptr[x];
    }
}

bool align2D(
        const cv::Mat& cur_img,
        uint8_t* ref_patch_with_border,
        uint8_t* ref_patch,
        const int n_iter,
        const bool affine_est_offset,
        const bool affine_est_gain,
        Eigen::Vector2d & cur_px_estimate,
        bool no_simd,
        std::vector<Eigen::Vector2f> *each_step)
{

    if(each_step) each_step->clear();

    const int halfpatch_size_ = 4;
    const int patch_size_ = 8;
    const int patch_area_ = 64;
    bool converged=false;

    // We optimize feature position and two affine parameters.
    // compute derivative of template and prepare inverse compositional
    float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
    float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
    Eigen::Matrix4f H; H.setZero();

    // compute gradient and hessian
    const int ref_step = patch_size_+2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    for(int y=0; y<patch_size_; ++y)
    {
        uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;//start form ref_patch_with_border(1,1)
        for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
        {
            Eigen::Vector4f J;
            J[0] = 0.5 * (it[1] - it[-1]);//ref_patch_with_border(2,1)-ref_patch_with_border(0,1);
            J[1] = 0.5 * (it[ref_step] - it[-ref_step]);//ref_patch_with_border(1,2)-ref_patch_with_border(1,0);

            // If not using the affine compensation, force the jacobian to be zero.
            J[2] = affine_est_offset? 1.0 : 0.0;
            J[3] = affine_est_gain? -1.0*it[0]: 0.0;

            *it_dx = J[0];
            *it_dy = J[1];
            H += J*J.transpose();
        }
    }
    // If not use affine compensation, force update to be zero by
    // * setting the affine parameter block in H to identity
    // * setting the residual block to zero (see below)
    if(!affine_est_offset)
    {
        H(2, 2) = 1.0;
    }
    if(!affine_est_gain)
    {
        H(3, 3) = 1.0;
    }
    Eigen::Matrix4f Hinv = H.inverse();
    float mean_diff = 0;
    float alpha = 1.0;

    // Compute pixel location in new image:
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    if(each_step) each_step->push_back(Eigen::Vector2f(u, v));

    // termination condition
    const float min_update_squared = 0.03*0.03; // TODO I suppose this depends on the size of the image (ate)
    const int cur_step = cur_img.step.p[0];
    //float chi2 = 0;
    Eigen::Vector4f update; update.setZero();
    for(int iter = 0; iter<n_iter; ++iter)
    {
        int u_r = std::floor(u);
        int v_r = std::floor(v);
        if(u_r < halfpatch_size_
           || v_r < halfpatch_size_
           || u_r >= cur_img.cols-halfpatch_size_
           || v_r >= cur_img.rows-halfpatch_size_)
            break;

        if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
            return false;

        // compute interpolation weights
        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;

        // loop through search_patch, interpolate
        uint8_t* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
        //float new_chi2 = 0.0;
        Eigen::Vector4f Jres; Jres.setZero();
        for(int y=0; y<patch_size_; ++y)
        {
            uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                float res = search_pixel - alpha*(*it_ref) + mean_diff;
                Jres[0] -= res*(*it_ref_dx);
                Jres[1] -= res*(*it_ref_dy);

                // If affine compensation is used,
                // set Jres with respect to affine parameters.
                if(affine_est_offset)
                {
                    Jres[2] -= res;
                }

                if(affine_est_gain)
                {
                    Jres[3] -= (-1)*res*(*it_ref);
                }
                //new_chi2 += res*res;
            }
        }
        // If not use affine compensation, force update to be zero.
        if(!affine_est_offset)
        {
            Jres[2] = 0.0;
        }
        if(!affine_est_gain)
        {
            Jres[3] = 0.0;
        }
        /*
        if(iter > 0 && new_chi2 > chi2)
        {
    #if SUBPIX_VERBOSE
          cout << "error increased." << endl;
    #endif
          u -= update[0];
          v -= update[1];
          break;
        }
        chi2 = new_chi2;
    */
        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];
        alpha += update[3];

        if(each_step) each_step->push_back(Eigen::Vector2f(u, v));


        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
            converged=true;
            break;
        }
    }

    cur_px_estimate << u, v;
    (void)no_simd;

    return converged;
}

int main() {
    //generate pattern in world
    std::vector<Eigen::Vector3d> circle;//in A frame,ref
    float r=5.0;
    for (int i = 0; i < 360; ++i) {
        Eigen::Vector3d p;
        p.z()=10;
        p.x()=r*cos(float(i)*M_PI/180.0);
        p.y()=r*sin(float(i)*M_PI/180.0);
        circle.push_back(p);
    }
    //cam intrinsic
    Eigen::Matrix3d K,Kinv;
    float fx=400,fy=400,cx=500,cy=500;
    K<<400.0,0,500.0,
            0,400.0,500.0,
            0,0,1.0;
    Kinv<<1.0,0,-500,
            0,1,-500,
            0,0,400.0;
    Kinv*=(1.0/400.0);
    int size=1000;
    //generate image at origin, aka, ref A
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
//frame B
    Eigen::Vector3d t(10,0,0);
    Eigen::AngleAxisd ry=Eigen::AngleAxisd(-M_PI/4.0,Eigen::Vector3d::UnitY());
    Eigen::Matrix4d T_BA_GT;
    T_BA_GT.setIdentity();
    T_BA_GT.topLeftCorner<3,3>()=ry.toRotationMatrix().transpose();
    T_BA_GT.topRightCorner<3,1>()= -ry.toRotationMatrix().transpose() * t;
    cv::Mat matB=cv::Mat::zeros(size,size,CV_8UC1);//cur
    //generate image at Frame B
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

    cv::Point2f srcTri[4];
    srcTri[0] = cv::Point2f( uv0px.x(),uv0px.y() );
    srcTri[1] = cv::Point2f( uv0mx.x(),uv0mx.y());
    srcTri[2] = cv::Point2f( uv0py.x(),uv0py.y() );
    srcTri[3] = cv::Point2f( uv0my.x(),uv0my.y() );
    cv::Point2f dstTri[4];
    dstTri[0] = cv::Point2f( uv0px_b.x(),uv0px_b.y());
    dstTri[1] = cv::Point2f( uv0mx_b.x(),uv0mx_b.y() );
    dstTri[2] = cv::Point2f( uv0py_b.x(),uv0py_b.y() );
    dstTri[3] = cv::Point2f( uv0my_b.x(),uv0my_b.y() );


    double n=0;
    n+=cv::norm(srcTri[0]-srcTri[2])/cv::norm(dstTri[0]-dstTri[2]);
    n+=cv::norm(srcTri[1]-srcTri[3])/cv::norm(dstTri[1]-dstTri[3]);
    std::cout<<"scale "<<n/2.0<<std::endl;

    cv::Mat warp_BA = cv::getPerspectiveTransform(  srcTri,dstTri );//A_ba
    std::cout<<" opencv 2 "<<std::endl;
    std::cout<<warp_BA<<std::endl;
    Eigen::Matrix<double,3,3> Wba;
    Wba.setZero();
    cv::cv2eigen(warp_BA,Wba);
//transform A to B
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
//    cv::imwrite("warpedB.jpg",warpedB);
//    cv::imwrite("A.jpg",matA);
//    cv::imwrite("B.jpg",matB);


    ImgPyr imgAPyr,imgBPyr;
    createImgPyramid(matA,4,imgAPyr);
    createImgPyramid(matB,4,imgBPyr);

    const int halfpatch_size=4;
    const int patch_size=halfpatch_size*2;

    if(0){
        //set initial guess,recall GT:
        //Eigen::Vector3d t(10,0,0);
        //Eigen::AngleAxisd ry=Eigen::AngleAxisd(-M_PI/4.0,Eigen::Vector3d::UnitY());
        Sophus::SE3d TAB_dis(Eigen::AngleAxisd(-M_PI / 4, Eigen::Vector3d::UnitY()).toRotationMatrix(), Eigen::Vector3d(10, 0, 0));
        Sophus::SE3d TBA_dis=TAB_dis.inverse();
        std::cout<<"ground truth"<<std::endl;
        std::cout << Sophus::SE3d(T_BA_GT).log().transpose() << std::endl;
        std::cout<<"before opt"<<std::endl;
        std::cout << TBA_dis.log().transpose() << std::endl;

        Sophus::SE3d TBA_back=TBA_dis;

        std::vector<int> valid_idx;
        Eigen::Matrix<double,6,6> H;
        Eigen::Matrix<double,6,1> B;
        H.setZero();
        B.setZero();
        bool init=true;

        for (int level = 3; level >=0; --level) {
            double scale=n/2.0;
            std::cout<<"current level "<<level<<std::endl;
            scale/=(1<<level);
            int search_level= std::log2(1.0/scale);
            if(search_level>3)
                search_level=3;
            std::cout<<"best search level "<<search_level<<std::endl;

            cv::Point2f srcLev[4];
            cv::Point2f dstLev[4];
            for (int i = 0; i < 4; ++i) {
                srcLev[i]=srcTri[i]/(1<<level);
                dstLev[i]=dstTri[i]/(1<<search_level);
            }
            cv::Mat warp_BA_level = cv::getPerspectiveTransform(  srcLev,dstLev );//A_ba
//        std::cout<<" opencv "<<std::endl;
//        std::cout<<warp_BA_level<<std::endl;

            Eigen::Matrix<double,3,3> Wlevel;
            Wlevel.setZero();
            cv::cv2eigen(warp_BA_level,Wlevel);



            std::vector<std::vector<Eigen::MatrixXd>> Js;
            std::vector<std::vector<float>> Res;
            Js.resize(circle.size());
            Res.resize(circle.size());
            bool converge=false;
            int max_iter=5;
            int iter=0;
            double last_sum_res=0;
            int idx=0;
            double lambda_vee=2;
            double min_lambda=1e-18;
            double lambda=1e-6;

            last_sum_res=linearize(circle, level, TBA_dis,Wlevel, K, search_level, patch_size, halfpatch_size, imgAPyr, imgBPyr, true, idx, Js, Res,valid_idx,init);
            init= false;
            std::cout<<"ave error "<<last_sum_res/float(idx*patch_size*patch_size)<<std::endl;

            H.setZero();
            B.setZero();
            for (int i = 0; i < valid_idx.size(); ++i) {
                for (int j = 0; j < patch_size*patch_size; ++j) {
                    H.noalias()+=Js[i][j].transpose()*Js[i][j];
                    B.noalias()-=Js[i][j].transpose()*Res[i][j];
                }
            }

            while(!(converge ||iter>max_iter)){
                Eigen::VectorXd Hdiag_lambda = Eigen::Matrix<double,6,1>::Ones()*lambda;
                for (int i = 0; i < Hdiag_lambda.size(); i++)
                    Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

                H.diagonal()+=Hdiag_lambda;

                TBA_back=TBA_dis;
                Eigen::Matrix<double,6,1> dx=H.ldlt().solve(B);
                TBA_dis*=Sophus::SE3d(Sophus::SO3d::exp(dx.tail<3>()),dx.head<3>());
                std::cout<<"iter: "<<iter<<" delta "<<dx.transpose()<<std::endl;
                if(abs(dx.maxCoeff())<1e-3){
                    converge=true;
                    std::cerr<<"converged "<<std::endl;
                    std::cout << TBA_dis.log().transpose() << std::endl;
                    break;
                }


                idx=0;
                double current_sum_res=linearize(circle, level, TBA_dis, Wlevel, K, search_level, patch_size, halfpatch_size, imgAPyr, imgBPyr,
                                                 false, idx, Js, Res,valid_idx,init);


                double f_diff = (last_sum_res - current_sum_res);
                double l_diff = 0.5 * dx.dot(dx * lambda + B);

                double step_quality = f_diff / l_diff;
                if(l_diff<0 || std::isnan(step_quality)){
                    step_quality=-1;
                    std::cerr<<"unknown error"<<std::endl;
                }

                if (step_quality < 0) {
                    lambda = std::min(100.0, lambda_vee * lambda);
                    lambda_vee *= 2;
                    TBA_dis=TBA_back;
                    std::cout << "\t[REJECTED] lambda:" << lambda
                              << " step_quality: " << step_quality
                              << " max_inc: " << dx.maxCoeff() << " Error: " << current_sum_res
                              << std::endl;
                    std::cout << TBA_dis.log().transpose() << std::endl;
                } else {
                    lambda = std::max(
                            min_lambda,
                            lambda *
                            std::max(1.0 / 3, 1 - std::pow(2 * step_quality - 1, 3.0)));
                    lambda_vee = 2;
                    std::cerr << "\t[ACCEPTED] lambda:" << lambda
                              << " step_quality: " << step_quality
                              << " max_inc: " << dx.maxCoeff() <<" Last Error: "<<last_sum_res<< " Error: " << current_sum_res<<std::endl;

                    last_sum_res=current_sum_res;

                    idx=0;
                    linearize(circle, level, TBA_dis,Wlevel, K, search_level, patch_size, halfpatch_size, imgAPyr, imgBPyr, true, idx, Js, Res,valid_idx,init);
                    H.setZero();
                    B.setZero();
                    for (int i = 0; i < valid_idx.size(); ++i) {
                        for (int j = 0; j < patch_size*patch_size; ++j) {
                            H+=Js[i][j].transpose()*Js[i][j];
                            B-=Js[i][j].transpose()*Res[i][j];
                        }
                    }
                }

                iter++;
            }

        }


        std::cout<<"ground truth"<<std::endl;
        std::cout << Sophus::SE3d(T_BA_GT).log().transpose() << std::endl;
        std::cout<<"after opt"<<std::endl;
        std::cout << TBA_dis.log().transpose() << std::endl;
    }












//only for debug

    if(0){
        for (int level = 3; level >=0; --level) {
            double scale=n/2.0;
            std::cout<<"current level "<<level<<std::endl;
            scale/=(1<<level);
            int search_level= std::log2(1.0/scale);
            if(search_level>3)
                search_level=3;
            std::cout<<"best search level "<<search_level<<std::endl;

            cv::Point2f srcLev[4];
            cv::Point2f dstLev[4];
            for (int i = 0; i < 4; ++i) {
                srcLev[i]=srcTri[i]/(1<<level);
                dstLev[i]=dstTri[i]/(1<<search_level);
            }
            cv::Mat warp_BA_level = cv::getPerspectiveTransform(  srcLev,dstLev );//A_ba
//        std::cout<<" opencv "<<std::endl;
//        std::cout<<warp_BA_level<<std::endl;

            Eigen::Matrix<double,3,3> Wlevel;
            Wlevel.setZero();
            cv::cv2eigen(warp_BA_level,Wlevel);

            std::vector<cv::Mat> ref_patches,cur_patches;

            for (int i = 0; i < circle.size(); i++) {
                Eigen::Vector3d pa = circle[i];
                pa /= pa.z();
                Eigen::Vector2d uva = (K * pa).topLeftCorner<2, 1>() / (1 << level);


                uint8_t A_patch_ptr[patch_size * patch_size]  __attribute__ ((aligned (16)));
                uint8_t *patch_ptr_a = A_patch_ptr;
                make_patch_with_warp(halfpatch_size, uva, imgAPyr[level], patch_ptr_a, Wlevel);//ref


                Eigen::Vector3d pb=ry.toRotationMatrix().transpose()*circle[i]-ry.toRotationMatrix().transpose()*t;
                Eigen::Vector2d uvb = (K * pb / pb.z()).topLeftCorner<2, 1>() / (1 << search_level);

                uint8_t B_patch_ptr[patch_size * patch_size]  __attribute__ ((aligned (16)));
                uint8_t *patch_ptr_b = B_patch_ptr;
                computeCurrentFeaturePatch(uvb, patch_size, imgBPyr[search_level], patch_ptr_b, nullptr, nullptr);


                double error=0;
                for (int j = 0; j < patch_size*patch_size; ++j) {
                    error+=fabs(patch_ptr_b[j]-patch_ptr_a[j]);
                }
                error/=double(patch_size*patch_size);

                cv::Mat patchA;
                patchToMat(patch_ptr_a,patch_size,&patchA);
                ref_patches.push_back(patchA);

                cv::Mat patchB;
                patchToMat(patch_ptr_b,patch_size,&patchB);
                cv::putText(patchB,std::to_string(int(error)),cv::Point(patch_size/2.0,patch_size/2.0),1,1,cv::Scalar::all(200));
                cur_patches.push_back(patchB);
            }
            cv::Mat debug_ref_img,debug_cur_img,full_imag;
            concatenatePatches(ref_patches,&debug_ref_img);
            concatenatePatches(cur_patches,&debug_cur_img);
            cv::vconcat(debug_ref_img,debug_cur_img,full_imag);

            cv::imwrite("level"+std::to_string(level)+"res.jpg",full_imag);
            cv::imwrite("level"+std::to_string(level)+"A.jpg",imgAPyr[level]);
            cv::imwrite("level"+std::to_string(level)+"B.jpg",imgBPyr[level]);


        }




    }

    std::vector<cv::KeyPoint> ref_pts,cur_pts;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < circle.size(); i+=10) {
        Eigen::Vector3d pa = circle[i];
        pa /= pa.z();
        Eigen::Vector2d uva = (K * pa).topLeftCorner<2, 1>();


        uint8_t A_patch_ptr_wb[(patch_size+1) * (patch_size+1)]  __attribute__ ((aligned (16)));
        uint8_t *patch_ptr_a_wb = A_patch_ptr_wb;

        uint8_t A_patch_ptr[(patch_size) * (patch_size)]  __attribute__ ((aligned (16)));
        uint8_t *patch_ptr_a= A_patch_ptr;

        make_patch_with_warp(halfpatch_size+1, uva, imgAPyr[0], patch_ptr_a_wb, Wba);//ref
        createPatchFromPatchWithBorder(patch_ptr_a_wb,patch_size,patch_ptr_a);

        Sophus::SE3d TAB_dis(Eigen::AngleAxisd(-M_PI / 4.1, Eigen::Vector3d::UnitY()).toRotationMatrix(), Eigen::Vector3d(10, 0, 0));
        Sophus::SE3d TBA_dis=TAB_dis.inverse();
        Eigen::Vector3d pb=TBA_dis*circle[i];
        Eigen::Vector2d uvb = (K * pb / pb.z()).topLeftCorner<2, 1>() ;

//        uint8_t B_patch_ptr[patch_size * patch_size]  __attribute__ ((aligned (16)));
//        uint8_t *patch_ptr_b = B_patch_ptr;
//        computeCurrentFeaturePatch(uvb, patch_size, imgBPyr[0], patch_ptr_b, nullptr, nullptr);
        bool res=align2D(matB,patch_ptr_a_wb,patch_ptr_a,10,true,false,uvb, false, nullptr);
        if(res){
            matches.emplace_back(ref_pts.size(),ref_pts.size(),1);
            ref_pts.emplace_back(uva.x(),uva.y(),1);
            cur_pts.emplace_back(uvb.x(),uvb.y(),1);

        }
    }
    cv::Mat res;
    cv::drawMatches(matA,ref_pts,matB,cur_pts,matches,res);
    cv::imwrite("res.jpg",res);

    return 0;
}
