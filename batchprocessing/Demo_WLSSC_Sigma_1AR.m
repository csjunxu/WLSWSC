clear;
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

% parameters
par.step = 3;       % the step of two neighbor patches
par.ps = 6;        % patch size
par.win = 20;   % size of window around the patch

par.outerIter = 4;
par.innerIter = 2;
par.WWIter = 100;
par.epsilon = 0.005;
par.model = 2;

for delta = 0
    par.delta = delta;
    for nSig0 = 0.1:0.05:0.2
        par.nSig0 = nSig0;
        for lambdasc = [0.01 0.1]
            par.lambdasc = lambdasc;
            PSNR = [];
            SSIM = [];
            CCPSNR = [];
            CCSSIM = [];
            for i = 1 : im_num
                par.nlsp = 70;  % number of non-local patches
                par.image = i;
                IMin = im2double(imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name) ));
                IM_GT = im2double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
                S = regexp(TT_im_dir(i).name, '\.', 'split');
                IMname = S{1};
                [h,w,ch] = size(IMin);
                fprintf('%s: \n', TT_im_dir(i).name);
                CCPSNR = [CCPSNR csnr( IMin*255,IM_GT*255, 0, 0 )];
                CCSSIM = [CCSSIM cal_ssim( IMin*255, IM_GT*255, 0, 0 )];
                fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', CCPSNR(end), CCSSIM(end));
                % read clean image
                par.I = IM_GT;
                par.nim = IMin;
                par.imIndex = i;
                t1=clock;
                [IMout, par]  =  WLSSC_Sigma_1A(par);
                t2=clock;
                etime(t2,t1)
                alltime(par.imIndex)  = etime(t2, t1);
                %% output
                PSNR = [PSNR csnr( IMout * 255, IM_GT * 255, 0, 0 )];
                SSIM = [SSIM cal_ssim( IMout * 255, IM_GT * 255, 0, 0 )];
                fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
                %% output
                %             imwrite(IMout, ['../cc_Results/Real_Offline/External_II_RGB_BID_' IMname '.png']);
            end
            mPSNR=mean(par.PSNR,2);
            [~, idx] = max(mPSNR);
            PSNR =par.PSNR(idx,:);
            SSIM = par.SSIM(idx,:);
            mSSIM=mean(SSIM,2);
            mtime  = mean(alltime);
            mCCPSNR = mean(CCPSNR);
            mCCSSIM = mean(CCSSIM);
            save(['WLSSC_Sigma_1AR_nSig' num2str(nSig0) '_delta' num2str(delta) '_lsc' num2str(lambdasc) '_WWIter' num2str(par.WWIter) '.mat'],'alltime','mtime','PSNR','mPSNR','SSIM','mSSIM','CCPSNR','mCCPSNR','CCSSIM','mCCSSIM');
        end
    end
end
