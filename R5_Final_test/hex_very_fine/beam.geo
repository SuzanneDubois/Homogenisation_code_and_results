// ============================================================
//  Tapered beam, length 20
//  Section 6 x 6 at x = 0  →  3 x 3 at x = Lx (linear taper)
//  Sections are centred on the axis (y, z) = (H0/2, H0/2)
//  Structured hex mesh, refinement toward x = Lx (x_max face)
// ============================================================

Lx = 20.0;
H0 = 6.0;   // section side at x = 0   (H0 x H0)
H1 = 1.5;   // section side at x = Lx  (H1 x H1)

c  = H0/2;  // axis position in y and z
a  = c - H1/2;
b  = c + H1/2;

Nx = 80;   // elements along X
Ny = 12;    // elements along Y
Nz = 12;    // elements along Z

p = 1;   // grading ratio — increase for stronger refinement at x=Lx

// ============================================================
//  POINTS
// ============================================================
// x = 0 face (6 x 6)
Point(1) = {0,  0,  0};
Point(4) = {0,  H0, 0};
Point(5) = {0,  0,  H0};
Point(8) = {0,  H0, H0};
// x = Lx face (3 x 3)
Point(2) = {Lx, a,  a};
Point(3) = {Lx, b,  a};
Point(6) = {Lx, a,  b};
Point(7) = {Lx, b,  b};

// ============================================================
//  CURVES
// ============================================================
// Along X — all oriented from x=0 to x=Lx
Line(1) = {1, 2};
Line(2) = {4, 3};
Line(3) = {5, 6};
Line(4) = {8, 7};
// Along Y
Line(5) = {1, 4};
Line(6) = {2, 3};
Line(7) = {5, 8};
Line(8) = {6, 7};
// Along Z
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {4, 8};
Line(12) = {3, 7};

// ============================================================
//  TRANSFINITE CURVES
//  X lines all go toward x=Lx → Progression p (refine at x=Lx)
// ============================================================
Transfinite Curve{1, 2, 3, 4}   = Nx + 1 Using Progression p;
Transfinite Curve{5, 6, 7, 8}   = Ny + 1 Using Progression 1.0;
Transfinite Curve{9, 10, 11, 12} = Nz + 1 Using Progression 1.0;

// ============================================================
//  SURFACES
// ============================================================
Curve Loop(1) = {1, 6, -2, -5};    Plane Surface(1) = {1};   // z_min
Curve Loop(2) = {3, 8, -4, -7};    Plane Surface(2) = {2};   // z_max
Curve Loop(3) = {1, 10, -3, -9};   Plane Surface(3) = {3};   // y_min
Curve Loop(4) = {2, 12, -4, -11};  Plane Surface(4) = {4};   // y_max
Curve Loop(5) = {5, 11, -7, -9};   Plane Surface(5) = {5};   // x_min
Curve Loop(6) = {6, 12, -8, -10};  Plane Surface(6) = {6};   // x_max

Transfinite Surface{1} = {1, 2, 3, 4};
Transfinite Surface{2} = {5, 6, 7, 8};
Transfinite Surface{3} = {1, 2, 6, 5};
Transfinite Surface{4} = {4, 3, 7, 8};
Transfinite Surface{5} = {1, 4, 8, 5};
Transfinite Surface{6} = {2, 3, 7, 6};
Recombine Surface{1:6};

// ============================================================
//  VOLUME
// ============================================================
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};
Transfinite Volume{1} = {1, 2, 3, 4, 5, 6, 7, 8};

// ============================================================
//  PHYSICAL GROUPS
// ============================================================
Physical Surface("z_min") = {1};
Physical Surface("z_max") = {2};
Physical Surface("y_min") = {3};
Physical Surface("x_max") = {6};
Physical Surface("y_max") = {4};
Physical Surface("x_min") = {5};

Physical Volume("solid") = {1};

// ============================================================
//  MESH OPTIONS
// ============================================================
Mesh.ElementOrder   = 1;
Mesh.RecombineAll   = 1;
Mesh.MshFileVersion = 2.2;
