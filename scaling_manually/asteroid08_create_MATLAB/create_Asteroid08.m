% Number of subdivisions
n = 100;

% Generate sphere
[X, Y, Z] = sphere(n);

% Convert surface to triangles
[F, V] = surf2patch(X, Y, Z, 'triangles');

% Create triangulation
TR = triangulation(F, V);

Vnew = TR.Points; 
Vnew(:,1:2) = Vnew(:,1:2)*1.24; 
TRnew = triangulation(TR.ConnectivityList, Vnew); 

% Write STL
stlwrite(TRnew, "Asteroid08.stl");