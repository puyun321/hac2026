% This MATLAB script modify Asteroid04 produced by convexinv to fit the
% scaling required by the organizers, assuming it is convex 

close all
clear all
clc

TR = stlread("2002NY14.stl");

theta = 95;    % degrees (xy-axis) 
vartheta = 118; % degree (xz-axis)  

R = [cosd(theta), -sind(theta), 0;
     sind(theta),  cosd(theta), 0;
     0,             0,          1];

R1 = [cosd(vartheta), 0, -sind(vartheta); 
     0,             1,          0;
     sind(vartheta), 0, cosd(vartheta)];

Vnew = (R * R1 * TR.Points')';
Vnew(:,3) = Vnew(:,3) - 0.02*ones(length(Vnew(:,3)),1); 

Vnew = Vnew/0.871; % scaling until z_min=-1 and z_max=1 
Vnew(:,1:2) = Vnew(:,1:2)*0.925/0.982; 

TRnew = triangulation(TR.ConnectivityList, Vnew);

stlwrite(TRnew, "Asteroid06.stl");
