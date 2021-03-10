# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    Nathan Bartlett <nbb0011@auburn.edu>
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticMesh import XicsrtOpticMesh

@dochelper
class XicsrtOpticAperture(XicsrtOpticMesh):
    """
    .. SourceNote
        See documentation or use help(XicsrtOpticCrystal) for a complete list
        of config options.
    """

    def default_config(self):
        """
        opt_size:  numpy array (None)
          The size of the actual optical elements. All rays hitting outside
          the optical element will be masked. 
          
        do_trace_local: bool (False)
          If true: transform rays to optic local coordinates before raytracing,
          do raytracing in local coordinates, then transform back to global
          coordinates.

          The default is 'false' as most built-in optics can perform raytracing
          in global coordinates. This option is convenient for optics with
          complex geometry for which intersection and reflection equations
          are easier or more clear to program in fixed local coordinates.

        do_miss_check: bool (true)
          Perform a check for whether the rays intersect the optic within the
          defined bounds (usually defined by 'width' and 'height'). If set to
          `False` all rays with a defined reflection/transmission condition
          will be traced.
        """
        config = super().default_config()
        
        # spactial information
        config['opt_size'] = None
        
        # boolean settings
        config['do_trace_local'] = True
        config['do_miss_check'] = True
        
        # mesh information
        config['use_meshgrid'] = False
        config['mesh_points'] = None
        config['mesh_faces'] = None
        config['mesh_normals'] = None
        config['mesh_coarse_points'] = None
        config['mesh_coarse_faces'] = None
        config['mesh_coarse_normals'] = None
        config['mesh_interpolate'] = True
        config['mesh_refine'] = None
                                                                              
        return config
    
    def initialize(self):
        super().initialize()
        
    def aperture_check(self,X,rays,m):
        '''
        Determines which rays will pass thorugh the aperture. Masks all others.
        '''
        if (self.param['opt_size'] == None).all():
            pass
        #circle apt
        elif (self.param['opt_size'].shape == (1,)):    
            m[m] &= (X[m,0]**2 + X[m,1]**2  < self.param['opt_size'][0]**2)    
        #square apt
        elif (self.param['opt_size'].shape == (2,)):    
            m[m] &= (np.abs(X[m,0]) < self.param['opt_size'][0] / 2)
            m[m] &= (np.abs(X[m,1]) < self.param['opt_size'][1] / 2)
        #complex apt
        else:
            mask_array = np.ones((len(self.param['opt_size']),len(m))) * m.T == 1
            for i in range(len(self.param['opt_size'])):
                #circ 
                if (self.param['opt_size'][i][2] == 0):   
                    mask_array[i][m] &= ((X[m,0] 
                                          - self.param['opt_size'][i][0])**2 
                                          + (X[m,1] - self.param['opt_size'][i][1])**2  
                                          < self.param['opt_size'][i][3]**2) 
                #square   
                elif (self.param['opt_size'][i][2] != 0):   
                    mask_array[i][m] &= (np.abs(X[m,0] 
                                         - self.param['opt_size'][i][0]) 
                                         < (self.param['opt_size'][i][2]) / 2)
                    mask_array[i][m] &= (np.abs(X[m,1] 
                                         - self.param['opt_size'][i][1]) 
                                         < (self.param['opt_size'][i][3]) / 2)        
            m[True] = mask_array[0]
            for i in range(len(mask_array)):
                # let ray pass through apt.    
                if self.param['opt_size'][i][4] == 1:
                    m[mask_array[i]] = True
                # block the ray at this location    
                elif self.param['opt_size'][i][4] == 0:
                    m[mask_array[i]] = False
            
            return
    
    
    
    
    
    
