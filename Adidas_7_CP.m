cd("E:\Image decomposition\Alignment_base_adidas_7_color\");

% Train
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_train.csv');
T(1,:)=[];

sam_mat_norm = [];
sam_mat_d1 = [];
sam_mat_d2 = [];
sam_mat_d3 = [];

sam_nmat_norm = [];
sam_nmat_d1 = [];
sam_nmat_d2 = [];
sam_nmat_d3 = [];

for i=1:size(T,1)

    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    if T{i,3}==0
        [dist_mat_1, dist_mat_2, dist_mat_3, norm_mat] = norm_dist(shape_1, shape_2);
        sam_mat_d1 = [sam_mat_d1 dist_mat_1];
        sam_mat_d2 = [sam_mat_d2 dist_mat_2];
        sam_mat_d3 = [sam_mat_d3 dist_mat_3];
        sam_mat_norm = [sam_mat_norm norm_mat];


    else
        [dist_nmat_1, dist_nmat_2, dist_nmat_3, norm_nmat] = norm_dist(shape_1, shape_2);
        sam_nmat_d1 = [sam_nmat_d1 dist_nmat_1];
        sam_nmat_d2 = [sam_nmat_d2 dist_nmat_2];
        sam_nmat_d3 = [sam_nmat_d3 dist_nmat_3];
        sam_mat_norm = [sam_mat_norm norm_nmat];
    end


    
end

% Validation
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_valid.csv');
T(1,:)=[];

sam_mat_norm_val = [];
sam_mat_d1_val = [];
sam_mat_d2_val = [];
sam_mat_d3_val = [];

sam_nmat_norm_val = [];
sam_nmat_d1_val = [];
sam_nmat_d2_val = [];
sam_nmat_d3_val = [];

for i=1:size(T,1)

    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    if T{i,3}==0
        [dist_mat_1, dist_mat_2, dist_mat_3, norm_mat] = norm_dist(shape_1, shape_2);
        sam_mat_d1_val = [sam_mat_d1_val dist_mat_1];
        sam_mat_d2_val = [sam_mat_d2_val dist_mat_2];
        sam_mat_d3_val = [sam_mat_d3_val dist_mat_3];
        sam_mat_norm_val = [sam_mat_norm_val norm_mat];


    else
        [dist_nmat_1, dist_nmat_2, dist_nmat_3, norm_nmat] = norm_dist(shape_1, shape_2);
        sam_nmat_d1_val = [sam_nmat_d1_val dist_nmat_1];
        sam_nmat_d2_val = [sam_nmat_d2_val dist_nmat_2];
        sam_nmat_d3_val = [sam_nmat_d3_val dist_nmat_3];
        sam_mat_norm_val = [sam_mat_norm_val norm_nmat];
    end


    
end



% Test
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_test.csv');
T(1,:)=[];

sam_mat_norm_test = [];
sam_mat_d1_test = [];
sam_mat_d2_test = [];
sam_mat_d3_test = [];

sam_nmat_norm_test = [];
sam_nmat_d1_test = [];
sam_nmat_d2_test = [];
sam_nmat_d3_test = [];

for i=1:size(T,1)

    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    if T{i,3}==0
        [dist_mat_1, dist_mat_2, dist_mat_3, norm_mat] = norm_dist(shape_1, shape_2);
        sam_mat_d1_test = [sam_mat_d1_test dist_mat_1];
        sam_mat_d2_test = [sam_mat_d2_test dist_mat_2];
        sam_mat_d3_test = [sam_mat_d3_test dist_mat_3];
        sam_mat_norm_test = [sam_mat_norm_test norm_mat];


    else
        [dist_nmat_1, dist_nmat_2, dist_nmat_3, norm_nmat] = norm_dist(shape_1, shape_2);
        sam_nmat_d1_test = [sam_nmat_d1_test dist_nmat_1];
        sam_nmat_d2_test = [sam_nmat_d2_test dist_nmat_2];
        sam_nmat_d3_test = [sam_nmat_d3_test dist_nmat_3];
        sam_mat_norm_test = [sam_mat_norm_test norm_nmat];
    end


    
end

