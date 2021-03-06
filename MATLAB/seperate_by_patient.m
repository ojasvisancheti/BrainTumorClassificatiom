clc;
clear;

% read every .mat file
mat = dir('*.mat'); 

for q = 1:length(mat) 
    fl = load(mat(q).name) 
    %x = fl.cjdata.PID
end
