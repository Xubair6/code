list = dir('*.mat');
mkdir HS_images
total_files = length(list);
filepath = '/HS_images/'
for i = 1:total_files
   
    f = load(list(i).name);
    
    outputDataCube = hypermnf(f.hypercube,3);
    save([fullfile('HS_images',list(i).name)], 'outputDataCube');
    %filename = fullfile(filepath, sprintf('test%d.mat',n));

end
