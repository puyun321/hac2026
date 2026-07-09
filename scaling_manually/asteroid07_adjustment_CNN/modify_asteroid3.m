% This MATLAB script modify Asteroid04 produced by convexinv to fit the
% scaling required by the organizers, assuming it is convex 

close all
clear all
clc

TR = stlread("asteroid3.stl");

theta = 46;    % degrees (xy-axis) 
vartheta = 55; % degree (xz-axis)  

R = [cosd(theta), -sind(theta), 0;
     sind(theta),  cosd(theta), 0;
     0,             0,          1];

R1 = [cosd(vartheta), 0, -sind(vartheta); 
     0,             1,          0;
     sind(vartheta), 0, cosd(vartheta)];

Vnew = (R1 * R * TR.Points')';

Vnew = Vnew/3.197; % scaling until z_min=-1 and z_max=1 
Vnew(:,1:2) = Vnew(:,1:2)*1.205/1.265; 

TRnew = triangulation(TR.ConnectivityList, Vnew);

stlwrite(TRnew, "Asteroid07.stl");
