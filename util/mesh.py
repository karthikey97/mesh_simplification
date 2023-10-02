import torch
import numpy as np
import numpy.linalg as la
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import TexturesUV
import heapq
import copy


OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path, boundary_weight=100):
        self.path = path
        mesh = load_objs_as_meshes([path])
        textures = mesh.textures

        verts = mesh.verts_packed().numpy()
        faces = mesh.faces_packed().numpy()

        ft = textures.faces_uvs_padded().numpy()[0,:,:]
        vt = textures.verts_uvs_padded().numpy()[0,:,:]
        self.texture_map = textures.maps_padded()[0]

        self.clean_mesh(faces, verts, ft, vt)
        
        self.verts = verts
        self.vn = mesh.verts_normals_packed()
        self.vs = np.concatenate([self.verts, self.vn], axis=1)
        self.faces = faces
        self.edges = mesh.edges_packed().numpy()
        ndim = self.vs.shape[1]

        self.A_s = [np.zeros((ndim,ndim)) for _ in range(len(self.vs))]
        self.B_s = [np.zeros((ndim,1)) for _ in range(len(self.vs))]
        self.C_s = [np.zeros((1,1)) for _ in range(len(self.vs))]

        self.build_v2v()
        self.build_vf()
        self.get_abcd()
        self.build_boundary_quadratics(boundary_weight)
        self.simp = False

    def clean_mesh(self, faces, verts, ft, vt):
        fvt = np.stack([faces,ft], axis=-1).reshape(-1, 2)
        fvs = np.concatenate([verts[faces], vt[ft]], axis=-1).reshape(fvt.shape[0], -1)

        unique_pairs, indx, inv = np.unique(fvt, axis=0, return_index=True, return_inverse=True)
        self.faces = inv.reshape(faces.shape)
        self.vs = fvs[indx]
        
        self.vert_mapping = [[] for _ in range(len(verts))]
        for i,new_vs in enumerate(unique_pairs[:,0]):
            self.vert_mapping[new_vs].append(i)
        
        edges, e2k, ec = [], {}, 0
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i+1)%3]]))
                if e not in e2k:
                    e2k[e] = ec
                    ec+=1
                    edges.append(list(e))
        self.edges = self.edges = np.array(edges, dtype=np.int64)

    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

    def build_v2v(self):
        v2v = [[] for _ in range(len(self.vs))]
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])
        self.v2v = v2v

    def get_abcd(self):
        self.p = self.vs[self.faces[:, 0]]
        self.e1 = self.vs[self.faces[:, 1]] - self.p
        self.e1 /= la.norm(self.e1, axis=1, keepdims=True) + 1e-24

        e2 = self.vs[self.faces[:, 2]] - self.p
        e2_p_e1 = (e2*self.e1).sum(axis=1, keepdims=True)*self.e1
        self.e2 = e2 - e2_p_e1
        self.e2 /= la.norm(self.e2, axis=1, keepdims=True) + 1e-24

    def build_boundary_quadratics(self, boundary_weight=100):
        ef = [[] for _ in range(len(self.edges))]
        boundary_edge_count = 0
        for i, (v0,v1) in enumerate(self.edges):
            ef[i] = self.vf[v0].intersection(self.vf[v1])
            if len(ef[i])==1:
                boundary_edge_count += 1
                f = self.faces[ef[i].pop()]
                v2 = (set(f).difference(set([v0,v1]))).pop()
                p,q,r = self.vs[v0], self.vs[v1], self.vs[v2]
                n = (r-p) - ((r-p)*(q-p)).sum()*(q-p)/la.norm(q-p)**2
                n /= la.norm(n)
                pn = (p*n).sum()
                A = boundary_weight*np.matmul(n[:,None],n[:,None].T)
                B = -boundary_weight*pn*n[:,None]
                C = boundary_weight*pn**2
                self.A_s[v0] += A
                self.A_s[v1] += A
                self.B_s[v0] += B
                self.B_s[v1] += B
                self.C_s[v0] += C
                self.C_s[v1] += C
                
    def simplification(self, target_v):
        vs, vf, edges = self.vs, self.vf, self.edges
        ndim = vs.shape[1]

        """ 1. compute Q for each vertex """
        for i, v in enumerate(vs):
            f_s = list(vf[i])
            for f in f_s:
                e1, e2, p = self.e1[f][:,None], self.e2[f][:,None], self.p[f][:,None]
                
                pe1 = np.sum(p * e1)
                pe2 = np.sum(p * e2)

                self.A_s[i] += np.eye(ndim)-np.matmul(e1, e1.T)-np.matmul(e2,e2.T)
                self.B_s[i] += e1*pe1 + e2*pe2 - p
                self.C_s[i] += np.matmul(p.T, p) - pe1**2 - pe2**2
            # E = np.matmul(v.T,np.matmul(self.A_s[i], v)) + 2*np.matmul(self.B_s[i].T, v) + self.C_s[i]
        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        inv_exists = 0
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            A_new, B_new, C_new = (self.A_s[e[0]]+self.A_s[e[1]]),(self.B_s[e[0]]+self.B_s[e[1]]),(self.C_s[e[0]]+self.C_s[e[1]])

            try:
                v_new = -np.matmul(la.inv(A_new), B_new)
                inv_exists += 1
            except:
                v_new = (0.5 * (v_0 + v_1))[:,None]

            E_new = np.matmul(v_new.T,np.matmul(A_new, v_new)) + 2*np.matmul(B_new.T, v_new) + C_new
            heapq.heappush(E_heap, (E_new[0,0], (e[0], e[1]), v_new[:,0], (A_new, B_new, C_new)))
        
        """ 3. collapse minimum-error vertex """
        print(f'Inverse exists for {inv_exists} edges out of {i+1}')
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]

        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1), v_new, Q_new = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                print('got boundary')
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                simp_mesh.vs[vi_0] = v_new
                simp_mesh.A_s[vi_0], simp_mesh.B_s[vi_0], simp_mesh.C_s[vi_0] = Q_new
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh

    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int64)] = False

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            A_new = (simp_mesh.A_s[vi_0]+simp_mesh.A_s[vv_i])
            B_new = (simp_mesh.B_s[vi_0]+simp_mesh.B_s[vv_i])
            C_new = (simp_mesh.C_s[vi_0]+simp_mesh.C_s[vv_i])
            try:
                v_mid = -np.matmul(la.inv(A_new), B_new).reshape(-1)
            except:
                v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])[:,None]

            E_new = np.matmul(v_mid.T,np.matmul(A_new, v_mid)) + 2*np.matmul(B_new.T, v_mid) + C_new
            heapq.heappush(E_heap, (E_new[0,0], (vi_0, vv_i), v_mid, (A_new, B_new, C_new)))

    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i
        
        simp_mesh.verts = simp_mesh.vs[:,:3]
        # simp_mesh.vt = simp_mesh.vs[:,3:5]

        # for vmap in simp_mesh.vert_mapping:
        #     v_corrected = np.zeros(3)
        #     c=0
        #     for v in vmap:
        #         v_corrected += simp_mesh.verts[vert_dict[v]]
        #         c+=1
        #     for v in vmap:
        #         simp_mesh.verts[vert_dict[v]] = v_corrected

        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        simp_mesh.verts = simp_mesh.verts[vi_mask]
        # simp_mesh.vt = simp_mesh.vt[vi_mask]
        simp_mesh.vs = simp_mesh.vs[vi_mask]

        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]

    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
        pool_hash = {}
        unpool_hash = {}
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        """ check """
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)

        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vetices cannot be covered!")
            return
        
        pool_hash = sorted(pool_hash.items(), key=lambda x:x[0])
        simp_mesh.pool_hash = pool_hash
        simp_mesh.unpool_hash = unpool_hash
            
    def save_(self,filename):
        vertices = torch.tensor(self.verts)
        verts_uvs = torch.tensor(self.vt.clip(0.,1.))
        faces = torch.tensor(self.faces)
        faces_uvs = torch.tensor(self.faces)
        save_obj(filename, verts=vertices, faces=faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=self.texture_map)
    
    def save(self, filename):
        assert len(self.vs) > 0
        vertices = np.array(self.verts, dtype=np.float32).flatten()
        # vt = np.array(self.vt, dtype=np.float32).flatten().clip(0., 1.)
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # for i in range(0, vt.size, 2):
            #     u = vertices[i + 0]
            #     v = vertices[i + 1]
            #     fp.write('vt {0:.8f} {1:.8f} {2:.8f}\n'.format(u, v))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(i0, i1, i2))
