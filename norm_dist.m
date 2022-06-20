function [dist1,dist2,dist3,norm] = norm_dist(shape1, shape2)
    exes_1 = tensor(shape1);
    exes_2 = tensor(shape2);
    
    r=1;
    dec_res_1 = cp_als(exes_1, r);
    dec_res_2 = cp_als(exes_2, r);
    
    shape1_1 = dec_res_1.U{1};
    shape1_2 = dec_res_1.U{2};
    shape1_3 = dec_res_1.U{3};
    shape2_1 = dec_res_2.U{1};
    shape2_2 = dec_res_2.U{2};
    shape2_3 = dec_res_2.U{3};

    dist1 = sqrt(sum((shape1_1 - shape2_1) .^ 2));
    dist2 = sqrt(sum((shape1_2 - shape2_2) .^ 2));
    dist3 = sqrt(sum((shape1_3 - shape2_3) .^ 2));
    norm = dist1 + dist2 + dist3;
end