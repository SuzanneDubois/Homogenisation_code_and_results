// ============================================================
//  Parallelepiped 20 x 3 x 3 — unstructured tetrahedral mesh
//  Optional refinement toward x = Lx (x_max face)
// ============================================================

Lx = 20.0;
Ly = 3.0;
Lz = 3.0;

h0 = 1.0;   // target element size at x = 0
h1 = 1.0;   // target element size at x = Lx (h1 < h0 refines toward x=Lx)

// ============================================================
//  POINTS  (base face z=0), with target mesh size
// ============================================================
Point(1) = {0,  0,  0, h0};
Point(2) = {Lx, 0,  0, h1};
Point(3) = {Lx, Ly, 0, h1};
Point(4) = {0,  Ly, 0, h0};

// ============================================================
//  CURVES
// ============================================================
Line(1) = {1, 2};   // bottom edge X
Line(2) = {2, 3};   // right edge  Y  (at x=Lx)
Line(3) = {3, 4};   // top edge    X
Line(4) = {4, 1};   // left edge   Y  (at x=0)

// ============================================================
//  SURFACE  (base face)
// ============================================================
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// ============================================================
//  EXTRUSION along Z  (geometry only: no Layers → free tet mesh)
//  extruded points keep the mesh size of the base points
// ============================================================
out[] = Extrude {0, 0, Lz} {
    Surface{1};
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
Mesh.Algorithm3D    = 1;   // Delaunay
Mesh.RecombineAll   = 0;   // keep triangles / tetrahedra
Mesh.MshFileVersion = 2.2;
