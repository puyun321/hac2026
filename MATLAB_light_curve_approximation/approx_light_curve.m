% This MATLAB script generates approximate asteroid light curves for the 
% Helsinki Asteroid Challenge 2026 (HAC2026) 

close all
clear all
clc

%% Inputs 
TR = stlread('Asteroid03.stl'); 

tau = 841; % discretization of angles for plotting light curve

% Placement of cameras 
angle1 = 0; % Placement of the first camera 
angle2 = 21*pi/180; % Placement of the second camera 

% The following table are placement of camera for HAC2026: 
% | no. | angle1 |   angle2 | 
% |   1 |      0 | 21pi/180 | 
% |   2 |   pi/4 | 26pi/180 | 
% |   3 |  2pi/4 | 26pi/180 | 
% |   4 |  3pi/4 | 26pi/180 | 
% |   5 |  5pi/4 | 24pi/180 | 
% |   6 |  6pi/4 | 24pi/180 | 
% |   7 |  7pi/4 | 24pi/180 | 

%% Main program 

% Vertices and faces
V = TR.Points;
F = TR.ConnectivityList;

% Unit normal vector of each face
N = faceNormal(TR);

% Coordinates of the three vertices of each face
P1 = V(F(:,1),:);
P2 = V(F(:,2),:);
P3 = V(F(:,3),:);

% Area of each triangular face
Area = 0.5 * vecnorm(cross(P2-P1, P3-P1, 2), 2, 2);


% Compute light curve 
direction = linspace(0,2*pi,tau); 
L1 = zeros(1,tau); 
L2 = zeros(1,tau); 

for k=1:tau
    E0 = transpose([cos(direction(k)), -sin(direction(k)), 0; sin(direction(k)), cos(direction(k)), 0; 0, 0, 1]*transpose([-1,0,0])); 
    E1 = transpose([cos(angle1+direction(k)), -sin(angle1+direction(k)), 0; sin(angle1+direction(k)), cos(angle1+direction(k)), 0; 0, 0, 1]*transpose([-1,0,0])); 
    E2 = transpose([cos(angle1+direction(k)), -sin(angle1+direction(k)), 0; sin(angle1+direction(k)), cos(angle1+direction(k)), 0; 0, 0, 1]*[cos(angle2), 0, sin(angle2); 0, 1, 0; -sin(angle2), 0, cos(angle2)]*transpose([-1,0,0])); 

    % Angles between normal of each face and 
    mu0 = max(N*transpose(E0),0); % incident angle of light source 
    mu1 = max(N*transpose(E1),0); % first camera 
    mu2 = max(N*transpose(E2),0); % second camera 

    % Lambert laws 
    S1 = mu0.*mu1; % for first camera 
    S2 = mu0.*mu2; % for second camera 

    % Light intensity 
    L1(1,k) = (transpose(Area)*S1); 
    L2(1,k) = (transpose(Area)*S2); 
end 

L1_normalized = L1/mean(L1); 
L2_normalized = L2/mean(L2); 

plot(direction,L1_normalized)
figure 
plot(direction,L2_normalized)


