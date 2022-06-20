cd("E:\Image decomposition\Alignment_base_adidas_7_color\");

% Train
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_train.csv');
T(1,:)=[];

x = [];
y = [];

for i=1:size(T,1)
    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    [x1 y1 x2 y2] = cp_x_y(shape_1, shape_2);
    x = [x x1 x2];
    y = [y y1 y2];
end

% Validation
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_valid.csv');
T(1,:)=[];

for i=1:size(T,1)
    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    [x1 y1 x2 y2] = cp_x_y(shape_1, shape_2);
    x = [x x1 x2];
    y = [y y1 y2];
end


% Test
T = readcell('E:\Image decomposition\Desktop\Adidas_7_pair_test.csv');
T(1,:)=[];

for i=1:size(T,1)
    shape_1 = double(imread(string(T{i,1})));
    shape_2 = double(imread(string(T{i,2})));

    [x1 y1 x2 y2] = cp_x_y(shape_1, shape_2);
    x = [x x1 x2];
    y = [y y1 y2];
end





