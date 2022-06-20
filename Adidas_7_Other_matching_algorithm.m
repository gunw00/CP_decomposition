cd("E:\Image decomposition\Alignment_base_adidas_7_color\");

% Train
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_train.csv');
T(1,:)=[];


SURF_mat = [];
SURF_feat_mat = [];
SIFT_mat = [];
SIFT_feat_mat = [];
BRISK_mat = [];
BRISK_feat_mat = [];
ORB_mat = [];
ORB_feat_mat = [];
Harris_mat = [];
Harris_feat_mat = [];
FAST_mat = [];
FAST_feat_mat = [];




for k=1:size(T,1)
    
    shape_1 = im2gray(imread(string(T{k,1})));
    shape_2 = im2gray(imread(string(T{k,2})));
    
    % SURF 256x1
    points_SURF_1 = detectSURFFeatures(shape_1);
    points_SURF_2 = detectSURFFeatures(shape_2);

    points_SURF_1_st10 = points_SURF_1.selectStrongest(10);
    points_SURF_2_st10 = points_SURF_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SURF_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SURF_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SURF_mat = [SURF_mat 0];
        SURF_feat_mat = [SURF_feat_mat 0];

    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        
        SURF_mat = [SURF_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SURF_feat_mat = [SURF_feat_mat sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end
    
    % SIFT 128x1
    points_SIFT_1 = detectSIFTFeatures(shape_1);
    points_SIFT_2 = detectSIFTFeatures(shape_2);
    
    points_SIFT_1_st10 = points_SIFT_1.selectStrongest(10);
    points_SIFT_2_st10 = points_SIFT_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SIFT_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SIFT_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SIFT_mat = [SIFT_mat 0];
        SIFT_feat_mat = [SIFT_feat_mat 0];
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        SIFT_mat = [SIFT_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SIFT_feat_mat = [SIFT_feat_mat sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end     

    % BRISK 64x1
    points_BRISK_1 = detectBRISKFeatures(shape_1);
    points_BRISK_2 = detectBRISKFeatures(shape_2);
    
    points_BRISK_1_st10 = points_BRISK_1.selectStrongest(10);
    points_BRISK_2_st10 = points_BRISK_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_BRISK_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_BRISK_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        BRISK_mat = [BRISK_mat 0];
        BRISK_feat_mat = [BRISK_feat_mat 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        BRISK_mat = [BRISK_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        BRISK_feat_mat = [BRISK_feat_mat sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % ORB 32x1
    points_ORB_1 = detectORBFeatures(shape_1);
    points_ORB_2 = detectORBFeatures(shape_2);
    
    points_ORB_1_st10 = points_ORB_1.selectStrongest(10);
    points_ORB_2_st10 = points_ORB_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_ORB_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_ORB_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;
    
    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        ORB_mat = [ORB_mat 0];
        ORB_feat_mat = [ORB_feat_mat 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        ORB_mat = [ORB_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        ORB_feat_mat = [ORB_feat_mat sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % Harris
    points_Harris_1 = detectHarrisFeatures(shape_1);
    points_Harris_2 = detectHarrisFeatures(shape_2);
    
    points_Harris_1_st10 = points_Harris_1.selectStrongest(10);
    points_Harris_2_st10 = points_Harris_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_Harris_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_Harris_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;


    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        Harris_mat = [Harris_mat 0];
        Harris_feat_mat = [Harris_feat_mat 0];
        
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        Harris_mat = [Harris_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        Harris_feat_mat = [Harris_feat_mat sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end       
    
    % FAST 64x1
    points_FAST_1 = detectFASTFeatures(shape_1);
    points_FAST_2 = detectFASTFeatures(shape_2);
    
    points_FAST_1_st10 = points_FAST_1.selectStrongest(10);
    points_FAST_2_st10 = points_FAST_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_FAST_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_FAST_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        FAST_mat = [FAST_mat 0];
        FAST_feat_mat = [FAST_feat_mat 0];
        
    else     

        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        FAST_mat = [FAST_mat sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        FAST_feat_mat = [FAST_feat_mat sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end        
end




% Validation
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_valid.csv');
T(1,:)=[];

SURF_mat_val = [];
SURF_feat_mat_val = [];
SIFT_mat_val = [];
SIFT_feat_mat_val = [];
BRISK_mat_val = [];
BRISK_feat_mat_val = [];
ORB_mat_val = [];
ORB_feat_mat_val = [];
Harris_mat_val = [];
Harris_feat_mat_val = [];
FAST_mat_val = [];
FAST_feat_mat_val = [];




for k=1:size(T,1)
    
    shape_1 = im2gray(imread(string(T{k,1})));
    shape_2 = im2gray(imread(string(T{k,2})));
    
    % SURF 256x1
    points_SURF_1 = detectSURFFeatures(shape_1);
    points_SURF_2 = detectSURFFeatures(shape_2);

    points_SURF_1_st10 = points_SURF_1.selectStrongest(10);
    points_SURF_2_st10 = points_SURF_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SURF_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SURF_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SURF_mat_val = [SURF_mat_val 0];
        SURF_feat_mat_val = [SURF_feat_mat_val 0];

    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        
        SURF_mat_val = [SURF_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SURF_feat_mat_val = [SURF_feat_mat_val sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end
    
    % SIFT 128x1
    points_SIFT_1 = detectSIFTFeatures(shape_1);
    points_SIFT_2 = detectSIFTFeatures(shape_2);
    
    points_SIFT_1_st10 = points_SIFT_1.selectStrongest(10);
    points_SIFT_2_st10 = points_SIFT_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SIFT_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SIFT_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SIFT_mat_val = [SIFT_mat_val 0];
        SIFT_feat_mat_val = [SIFT_feat_mat_val 0];
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        SIFT_mat_val = [SIFT_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SIFT_feat_mat_val = [SIFT_feat_mat_val sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end     

    % BRISK 64x1
    points_BRISK_1 = detectBRISKFeatures(shape_1);
    points_BRISK_2 = detectBRISKFeatures(shape_2);
    
    points_BRISK_1_st10 = points_BRISK_1.selectStrongest(10);
    points_BRISK_2_st10 = points_BRISK_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_BRISK_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_BRISK_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        BRISK_mat_val = [BRISK_mat_val 0];
        BRISK_feat_mat_val = [BRISK_feat_mat_val 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        BRISK_mat_val = [BRISK_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        BRISK_feat_mat_val = [BRISK_feat_mat_val sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % ORB 32x1
    points_ORB_1 = detectORBFeatures(shape_1);
    points_ORB_2 = detectORBFeatures(shape_2);
    
    points_ORB_1_st10 = points_ORB_1.selectStrongest(10);
    points_ORB_2_st10 = points_ORB_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_ORB_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_ORB_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;
    
    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        ORB_mat_val = [ORB_mat_val 0];
        ORB_feat_mat_val = [ORB_feat_mat_val 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        ORB_mat_val = [ORB_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        ORB_feat_mat_val = [ORB_feat_mat_val sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % Harris
    points_Harris_1 = detectHarrisFeatures(shape_1);
    points_Harris_2 = detectHarrisFeatures(shape_2);
    
    points_Harris_1_st10 = points_Harris_1.selectStrongest(10);
    points_Harris_2_st10 = points_Harris_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_Harris_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_Harris_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;


    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        Harris_mat_val = [Harris_mat_val 0];
        Harris_feat_mat_val = [Harris_feat_mat_val 0];
        
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        Harris_mat_val = [Harris_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        Harris_feat_mat_val = [Harris_feat_mat_val sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end       
    
    % FAST 64x1
    points_FAST_1 = detectFASTFeatures(shape_1);
    points_FAST_2 = detectFASTFeatures(shape_2);
    
    points_FAST_1_st10 = points_FAST_1.selectStrongest(10);
    points_FAST_2_st10 = points_FAST_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_FAST_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_FAST_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        FAST_mat_val = [FAST_mat_val 0];
        FAST_feat_mat_val = [FAST_feat_mat_val 0];
        
    else     

        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        FAST_mat_val = [FAST_mat_val sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        FAST_feat_mat_val = [FAST_feat_mat_val sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end        
end


% Test
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_test.csv');
T(1,:)=[];

SURF_mat_test = [];
SURF_feat_mat_test = [];
SIFT_mat_test = [];
SIFT_feat_mat_test = [];
BRISK_mat_test = [];
BRISK_feat_mat_test = [];
ORB_mat_test = [];
ORB_feat_mat_test = [];
Harris_mat_test = [];
Harris_feat_mat_test = [];
FAST_mat_test = [];
FAST_feat_mat_test = [];




for k=1:size(T,1)
    
    shape_1 = im2gray(imread(string(T{k,1})));
    shape_2 = im2gray(imread(string(T{k,2})));
    
    % SURF 256x1
    points_SURF_1 = detectSURFFeatures(shape_1);
    points_SURF_2 = detectSURFFeatures(shape_2);

    points_SURF_1_st10 = points_SURF_1.selectStrongest(10);
    points_SURF_2_st10 = points_SURF_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SURF_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SURF_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SURF_mat_test = [SURF_mat_test 0];
        SURF_feat_mat_test = [SURF_feat_mat_test 0];

    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        
        SURF_mat_test = [SURF_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SURF_feat_mat_test = [SURF_feat_mat_test sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end
    
    % SIFT 128x1
    points_SIFT_1 = detectSIFTFeatures(shape_1);
    points_SIFT_2 = detectSIFTFeatures(shape_2);
    
    points_SIFT_1_st10 = points_SIFT_1.selectStrongest(10);
    points_SIFT_2_st10 = points_SIFT_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_SIFT_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_SIFT_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        SIFT_mat_test = [SIFT_mat_test 0];
        SIFT_feat_mat_test = [SIFT_feat_mat_test 0];
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        SIFT_mat_test = [SIFT_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        SIFT_feat_mat_test = [SIFT_feat_mat_test sum(sqrt(sum((f1(paired_idx(:,1),:)-f2(paired_idx(:,2),:)).'.^2)))];
    end     

    % BRISK 64x1
    points_BRISK_1 = detectBRISKFeatures(shape_1);
    points_BRISK_2 = detectBRISKFeatures(shape_2);
    
    points_BRISK_1_st10 = points_BRISK_1.selectStrongest(10);
    points_BRISK_2_st10 = points_BRISK_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_BRISK_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_BRISK_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        BRISK_mat_test = [BRISK_mat_test 0];
        BRISK_feat_mat_test = [BRISK_feat_mat_test 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        BRISK_mat_test = [BRISK_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        BRISK_feat_mat_test = [BRISK_feat_mat_test sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % ORB 32x1
    points_ORB_1 = detectORBFeatures(shape_1);
    points_ORB_2 = detectORBFeatures(shape_2);
    
    points_ORB_1_st10 = points_ORB_1.selectStrongest(10);
    points_ORB_2_st10 = points_ORB_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_ORB_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_ORB_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;
    
    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        ORB_mat_test = [ORB_mat_test 0];
        ORB_feat_mat_test = [ORB_feat_mat_test 0];
        
    else 
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end

        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        ORB_mat_test = [ORB_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        ORB_feat_mat_test = [ORB_feat_mat_test sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end
    
    % Harris
    points_Harris_1 = detectHarrisFeatures(shape_1);
    points_Harris_2 = detectHarrisFeatures(shape_2);
    
    points_Harris_1_st10 = points_Harris_1.selectStrongest(10);
    points_Harris_2_st10 = points_Harris_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_Harris_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_Harris_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;


    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        Harris_mat_test = [Harris_mat_test 0];
        Harris_feat_mat_test = [Harris_feat_mat_test 0];
        
    else
        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        Harris_mat_test = [Harris_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        Harris_feat_mat_test = [Harris_feat_mat_test sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end       
    
    % FAST 64x1
    points_FAST_1 = detectFASTFeatures(shape_1);
    points_FAST_2 = detectFASTFeatures(shape_2);
    
    points_FAST_1_st10 = points_FAST_1.selectStrongest(10);
    points_FAST_2_st10 = points_FAST_2.selectStrongest(10);
    
    [f1,vpts1] = extractFeatures(shape_1,points_FAST_1_st10);
    [f2,vpts2] = extractFeatures(shape_2,points_FAST_2_st10);

    paired_idx = [];
    loc1=vpts1.Location;
    loc2=vpts2.Location;

    if length(loc1(:,1))==0 | length(loc2(:,1))==0
        FAST_mat_test = [FAST_mat_test 0];
        FAST_feat_mat_test = [FAST_feat_mat_test 0];
        
    else     

        for i=1:length(loc1(:,1))
            dist_norm = [];
            for j=1:length(loc2(:,1))
                dist_norm = [dist_norm norm(loc1(i,:)-loc2(j,:))];
            end
            [M, I] = min(dist_norm);
            paired_idx = [paired_idx ; i I];
            loc2(I,:) = 0;
        end
        
        matchedPoints1 = vpts1(paired_idx(:,1));
        matchedPoints2 = vpts2(paired_idx(:,2));
        
        FAST_mat_test = [FAST_mat_test sum(sqrt(sum((matchedPoints1.Location-matchedPoints2.Location).'.^2)))];
        FAST_feat_mat_test = [FAST_feat_mat_test sum(sqrt(sum((f1.Features(paired_idx(:,1),:)-f2.Features(paired_idx(:,2),:)).'.^2)))];
    end        
end

