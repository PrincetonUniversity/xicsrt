# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>
   
   Method-1 : Solving Quartic Equation by constructing companion matrix and finding it's EigenValue

               A * t ^ 4 + B * t ^ 3 + C * t ^ 2 + D * t + E = 0, which is then normalized 

               (A * t ^ 4 + B * t ^ 3 + C * t ^ 2 + D * t + E) * (1 / A) = 0

               t ^ 4 + a * t ^ 3 + b * t ^ 2 + c * t + d = 0

               Companion Matrix is,
               M = [[0,0,0,-d],
                    [1,0,0,-c],
                    [0,1,0,-b],
                    [0,0,1,-a]]
        
   reference : https://nkrvavica.github.io/post/on_computing_roots/
"""
import numpy as np

from multiQuartic import eig_quartic_roots          # Multiple Quartic Solver with EigenValue & Matrix Approach

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

@dochelper
class XicsrtOpticCrystalToroidal(XicsrtOpticCrystal):

    def default_config(self):
        config = super().default_config()
        
        """
        Rmajor:
                Major Radius of the Torus
        Rminor:
                Minor Radius of the Torus
        concave:
                If True it will consider intersection of Torus concave surface with Rays only, otherwise 
                it will consider intersection of Torus convex surface with Rays only
        inside:
                If True it will consider intersection of Torus which will give reflection from inside of
                the Torus Tube otherwise it will consider intersection of Torus which will give reflection
                from outside of the Torus Tube
        """
        
        config['Rmajor']  = 1.1
        config['Rminor']  = 0.2
        config['concave'] = True
        config['inside']  = False
        
        return config

    def initialize(self):
        super().initialize()
        
        """
            Here, we considered Torus to be buit around y-axis, so we construct torus axis from
            system axes given such as torus z-axis goes in yaxis of the system.
            And system z-axis & x-axis goes to torus x-axis and y-axis respectively
            
            Torus center is defined as,
                Torus_Center = Torus_Major_Radius * Torus_X_axis + System_Origin
                Here, Torus_X_axis = System_Z_axis
        """
        
        self.torusXaxis = self.param['zaxis']
        self.torusYaxis = self.param['xaxis']
        self.torusZaxis = np.cross(syszaxis, sysxaxis)

        self.param['center'] = self.param['Rmajor'] * self.torusXaxis + self.param['origin']
    
    def intersect(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        Toroidal optic.
        """
        
        # setup
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        Rmajor = self.param['Rmajor']
        Rminor = self.param['Rminor']  
        
        # variable setup
        distances = np.zeros(masks.shape, dtype=np.float64)
        
        orig = O - self.param['Center']      

        # Calculaing Ray Direction components in Torus coordinate system (Transforming Ray)
        d = np.zeros((len(m),3), dtype= np.float64)
        d[:, 0] = np.dot(D, torusXaxis)
        d[:, 1] = np.dot(D, torusYaxis)
        d[:, 2] = np.dot(D, torusZaxis)

        # Calculaing Ray Origin components in Torus coordinate system (Transforming Ray Origin)
        dOrig = np.zeros((len(m),3), dtype= np.float64)
        dOrig[:, 0] = np.dot(orig, torusXaxis)
        dOrig[:, 1] = np.dot(orig, torusYaxis)
        dOrig[:, 2] = np.dot(orig, torusZaxis)
        
        # Calculaing Magnitude of Ray Direction
        dMag = np.sqrt(np.einsum('ij,ij->i', d, d))s
        
        # defining resusable variables
        distRayOrigin2OriginSq = np.einsum('ij,ij->i', dOrig, dOrig)
        rayCompDirOnOrigin = np.einsum('ij,ij->i', dOrig, d)

        R1 = Rmajor ** 2 + Rminor ** 2 

        
        """
            The form of quartic equation to be solved is,
            c0 ** t ^ 4 + c1 ** t ^ 3 + c2 ** t ^ 2 + c3 ** t + c4 = 0
        """
        
        # defining co-efficients of Quartic Equation
        c0 = dMag ** 4
        c1 = 4 * dMag ** 2 * rayCompDirOnOrigin
        c2 = 4 * rayCompDirOnOrigin ** 2 + 2 * distRayOrigin2OriginSq * dMag ** 2 - 2 * R1 * dMag ** 2 + 4 * Rmajor ** 2 * d[:, 2] ** 2
        c3 = 4 * rayCompDirOnOrigin * (distRayOrigin2OriginSq - R1) + 8 * Rmajor ** 2 * d[:, 2] * dOrig[:, 2]
        c4 = distRayOrigin2OriginSq ** 2 - 2 * R1 * distRayOrigin2OriginSq + 4 * Rmajor ** 2 * dOrig[:, 2] ** 2 + (Rmajor ** 2 - Rminor ** 2) ** 2

        roots = eig_quartic_roots([c0, c1, c2, c3, c4])
       
        #    neglecting complex solution of the quartic equation       
        roots[roots != np.conjugate(roots)] = 0.0

        # considering user requirement of concave or convex surface intersection
        roots1 = np.zeros((len(m),2), dtype=np.complex64)
        roots1[:,0] = roots[:, (2 if params['concave'] else 0)]
        roots1[:,1] = roots[:, (3 if params['concave'] else 1)]    

        # Also considering user requirement of inside reflection or outside reflection
        distances[m] = np.where((roots1[m,0] > roots1[m,1]) if params['inside'] else (roots1[m,0] < roots1[m,1]), roots1[m,0],roots1[m,1])

        # Filtering out only positive roots
        m &= (distances[m] > 0.0)
        return distances
        
    
    # Generates normals
    def generate_normals(self, X, rays):
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        
        """
            Here, we first translates torus to the origin, then translates the circle on which 
            the point lies to the origin and then radius vector at that point will give normal
            at that point
        """
        
        # Simulates the Translation of Torus to the Origin
        pt = np.subtract(X[m], self.param['center'])
        
        """
        Checks that if Ray is intersecting Torus from back-side
        
        If ray is from back-side, ray-direction will give (+ve) component along the vector 
        from the Center of torus to the given point, otherwise, it should give (-ve) component.
        
        And, If ray is intersecting from back-side then the normal at intersection point should be flipped
        
            fromback = np.einsum('ij,j->i', pt, rays['direction']) > 0
            
        NOTE:
            In order to match with the sphere case, this calculation is not performed
        """
    
        """
            Calculates the Center of the Circle on which the point lies,
            by subtracting which the circle goes to the origin 
        """
        pt1 = np.subtract(pt, np.einsum('ij,j->i', pt, self.torusZaxis) * self.torusZaxis)
        pt1 = Rmajor * pt1 / p.sqrt(np.einsum('ij,ij->i', pt1, pt1))
        
        """
        Checks that if Ray will reflect from inside or outside of the Torus tube
        
        If ray is from inside, ray-direction will give (+ve) component along the vector 
        from the Center of circle to the given point, otherwise, it should give (-ve) component.
        
        And, If ray is intersecting from inside then the normal at intersection point should be flipped
        
            inside = np.einsum('ij,j->i', pt, pt - pt1) < 0
        
        NOTE:
            In order to match with the sphere case, this calculation is not performed
        """        
        
        # Simulates the circle having the intersection point at the origin And getting radius vector
        pt1 = np.subtract(pt, pt1)
        pt1 = pt1 / p.sqrt(np.einsum('ij,ij->i', pt1, pt1))
        
        """            
        pt1[fromback] = -pt1[fromback]      # flipping normal if ray is hitting from back-side
        pt1[inside] = -pt1[inside]          # flipping normal if point is on the inside ring of torus
                
        NOTE:
            In order to match with the spherical case, this calculation is not performed
        """
        normals[m] = pt1
        
        return normals