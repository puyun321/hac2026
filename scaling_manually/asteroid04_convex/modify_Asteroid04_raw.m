% This MATLAB script modify Asteroid04 produced by convexinv to fit the
% scaling required by the organizers, assuming it is convex 

close all
clear all
clc

TR = stlread("Asteroid04_raw.stl");

theta = 75;    % degrees (xy-axis) 
vartheta = 5; % degree (xz-axis)  

R = [cosd(theta), -sind(theta), 0;
     sind(theta),  cosd(theta), 0;
     0,             0,          1];

R1 = [cosd(vartheta), 0, -sind(vartheta); 
     0,             1,          0;
     sind(vartheta), 0, cosd(vartheta)];

Vnew = (R1 * R * TR.Points')';

Vnew = Vnew/10.725; % scaling until z_min=-1 and z_max=1 
Vnew(:,1:2) = Vnew(:,1:2)*1.475/1.168; 

TRnew = triangulation(TR.ConnectivityList, Vnew);

stlwrite(TRnew, "Asteroid04.stl");
