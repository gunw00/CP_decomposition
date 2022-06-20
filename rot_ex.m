clear;

shape = double(imread('C:\Users\gunwu\Dropbox\PC\Desktop\Stat\master\CP decomp\Rotated Data\160197L_20180502_2_2_2_jekruse_rot.png'));
exes = tensor(shape);




for r=[1,2,3,5,10]
	dec_res = cp_als(exes, r);
	
    figure;
    imagesc(double(full(dec_res))/255);
    colormap(gray);
    set(gca,'visible','off');
    saveas(gcf, ['Decomp_Rotated_data/D2_rot_r', num2str(r)], 'png');
    r
end