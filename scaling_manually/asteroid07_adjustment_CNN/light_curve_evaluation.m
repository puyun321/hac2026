% This MATLAB script generates approximate asteroid light curves for the 
% Helsinki Asteroid Challenge 2026 (HAC2026) 
% The main purpose of this program is to verify the reconstructed asteroid
% using light curves 

close all
clear all
clc

%% Inputs 
filename = 'Asteroid07.stl'; 
TR = stlread(filename); 
lightcurve_data = readmatrix('Asteroid07_lightcurve_intensity.csv'); 

tau = size(lightcurve_data,1); % discretization of angles for plotting light curve


%% Main program 

% Vertices and faces
V = TR.Points;
F = TR.ConnectivityList;

% Verify that the scaling is correct 
z_max = max(V(:,3)); 
z_min = min(V(:,3)); 
R_bound = max(sqrt(V(:,1).^2 + V(:,2).^2)); 

% Unit normal vector of each face
N = faceNormal(TR);

% Coordinates of the three vertices of each face
P1 = V(F(:,1),:);
P2 = V(F(:,2),:);
P3 = V(F(:,3),:);

% Area of each triangular face
Area = 0.5 * vecnorm(cross(P2-P1, P3-P1, 2), 2, 2);


% Placement of cameras 
angle1 = [0, pi/4, 2*pi/4, 3*pi/4, 5*pi/4, 6*pi/4, 7*pi/4]; % Placement of the first camera 
angle2 = [21*pi/180, 26*pi/180, 26*pi/180, 26*pi/180, 24*pi/180, 24*pi/180, 24*pi/180]; % Placement of the second camera 

% The following table are placement of camera for HAC2026: 
% | no. | angle1 |   angle2 | 
% |   1 |      0 | 21pi/180 | 
% |   2 |   pi/4 | 26pi/180 | 
% |   3 |  2pi/4 | 26pi/180 | 
% |   4 |  3pi/4 | 26pi/180 | 
% |   5 |  5pi/4 | 24pi/180 | 
% |   6 |  6pi/4 | 24pi/180 | 
% |   7 |  7pi/4 | 24pi/180 | 

% Compute light curve for each experiment 
direction = linspace(0,2*pi,tau); 
L1 = zeros(7,tau); 
L2 = zeros(7,tau); 
L1_normalized = zeros(7,tau); 
L2_normalized = zeros(7,tau); 

for a=1:7
  for k=1:tau
    E0 = transpose([cos(direction(k)), -sin(direction(k)), 0; sin(direction(k)), cos(direction(k)), 0; 0, 0, 1]*transpose([-1,0,0])); 
    E1 = transpose([cos(angle1(a)+direction(k)), -sin(angle1(a)+direction(k)), 0; sin(angle1(a)+direction(k)), cos(angle1(a)+direction(k)), 0; 0, 0, 1]*transpose([-1,0,0])); 
    E2 = transpose([cos(angle1(a)+direction(k)), -sin(angle1(a)+direction(k)), 0; sin(angle1(a)+direction(k)), cos(angle1(a)+direction(k)), 0; 0, 0, 1]*[cos(angle2(a)), 0, sin(angle2(a)); 0, 1, 0; -sin(angle2(a)), 0, cos(angle2(a))]*transpose([-1,0,0])); 

    % Angles between normal of each face and 
    mu0 = max(N*transpose(E0),0); % incident angle of light source 
    mu1 = max(N*transpose(E1),0); % first camera 
    mu2 = max(N*transpose(E2),0); % second camera 

    % Lambert laws 
    S1 = mu0.*mu1; % for first camera 
    S2 = mu0.*mu2; % for second camera 

    % Binary curve 
    L1(a,k) = transpose(Area)*S1; 
    L2(a,k) = transpose(Area)*S2; 
  end 
  L1_normalized(a,:) = L1(a,:)/mean(L1(a,:)); 
  L2_normalized(a,:) = L2(a,:)/mean(L2(a,:)); 
  
  subplot(7,2,2*a-1) 
  plot(direction,L1_normalized(a,:),'b')
  hold on 
  plot(direction,transpose(lightcurve_data(:,4*a-2)),'r') 
  legend('Reconstructed data', 'Original data', 'Location', 'best')
  title(['first camera, experiment no ', num2str(a)])

  subplot(7,2,2*a) 
  plot(direction,L2_normalized(a,:),'b')
  hold on 
  plot(direction,transpose(lightcurve_data(:,4*a)),'r') 
  legend('Reconstructed data', 'Original data', 'Location', 'best')
  title(['second camera, experiment no ', num2str(a)]) 
end 

sgtitle(sprintf('Evaluation of %s\nz_{min} = %.3f, z_{max} = %.3f, R_{bound} = %.3f', filename, z_min, z_max, R_bound));