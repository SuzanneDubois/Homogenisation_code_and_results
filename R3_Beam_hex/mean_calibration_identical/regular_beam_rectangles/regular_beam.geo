// ============================================================
//  Parallelepiped 20 x 3 x 3 — structured hex mesh
//  Refinement toward x = Lx (x_max face)
// ============================================================

Lx = 20.0;
Ly = 3.0;
Lz = 3.0;

Nx = 10;   // elements along X
Ny = 3;    // elements along Y
Nz = 3;    // elements along Z

p = 1;   // grading ratio — increase for stronger refinement at x=Lx

// ============================================================
//  POINTS  (base face z=0)
// ============================================================
Point(1) = {0,  0,  0};
Point(2) = {Lx, 0,  0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0,  Ly, 0};

// ============================================================
//  CURVES
// ============================================================
Line(1) = {1, 2};   // bottom edge X  → ends at x=Lx ★
Line(2) = {2, 3};   // right edge  Y  (at x=Lx, uniform)
Line(3) = {3, 4};   // top edge    X  → starts at x=Lx ★
Line(4) = {4, 1};   // left edge   Y  (at x=0, uniform)

// ============================================================
//  SURFACE  (base face)
// ============================================================
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 3, 4};
Recombine Surface{1};

// ============================================================
//  TRANSFINITE CURVES
//  Line 1: 1→2, goes toward x=Lx → Progression p (refine at end)
//  Line 3: 3→4, starts at x=Lx  → Progression 1/p (refine at start)
//  Lines 2,4: Y direction, uniform
// ============================================================
Transfinite Curve{1} = Nx + 1   Using Progression p;    // 1→2, refine at x=Lx
Transfinite Curve{3} = Nx + 1   Using Progression 1/p;  // 3→4, refine at x=Lx
Transfinite Curve{2} = Ny + 1   Using Progression 1.0;  // Y, uniform
Transfinite Curve{4} = Ny + 1   Using Progression 1.0;  // Y, uniform

// ============================================================
//  EXTRUSION along Z  (uniform)
// ============================================================
out[] = Extrude {0, 0, Lz} {
    Surface{1};
    Layers{Nz};
    Recombine;
};

// ============================================================
//  PHYSICAL GROUPS
// ============================================================
Physical Surface("z_min") = {1};
Physical Surface("z_max") = {out[0]};
Physical Surface("y_min") = {out[2]};
Physical Surface("x_max") = {out[3]};
Physical Surface("y_max") = {out[4]};
Physical Surface("x_min") = {out[5]};

Physical Volume("solid") = {out[1]};

// ============================================================
//  MESH OPTIONS
// ============================================================
Mesh.ElementOrder   = 1;
Mesh.RecombineAll   = 1;
Mesh.MshFileVersion = 2.2;
