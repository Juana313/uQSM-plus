import numpy as np
import os, glob, shutil
from time import gmtime, strftime
from datetime import datetime
import nibabel as nib


config = {}

config['saveFolder'] = r'./fb_roi_96x96x96'
config['mat_size'] = [96, 96, 96]
config['voxel_size'] = [1, 1, 1.0]

config['count'] = 100

def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
    
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)

def readNifti(fileName):
    if not os.path.exists(fileName):
        return None, None
    img = nib.load(fileName)
    return img.get_data(), img.affine

def Ellipsoid(ellipsoid, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (ellipsoid.shape[0]):
        A   = ellipsoid[k,0]
        asq = ellipsoid[k,1]**2
        bsq = ellipsoid[k,2]**2
        csq = ellipsoid[k,3]**2
        x0  = ellipsoid[k,4]
        y0  = ellipsoid[k,5]
        z0  = ellipsoid[k,6]
        phi = ellipsoid[k,7]
        theta = ellipsoid[k,8]
        psi   = ellipsoid[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = (((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq + (coordp[2,:]-z0)**2/csq) <= 1)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
        
    return p

def Cylinder(cylinder, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (cylinder.shape[0]):
        A   = cylinder[k,0]
        asq = cylinder[k,1]**2
        bsq = cylinder[k,2]**2
        zl  = cylinder[k,3]
        x0  = cylinder[k,4]
        y0  = cylinder[k,5]
        z0  = cylinder[k,6]
        phi = cylinder[k,7]
        theta = cylinder[k,8]
        psi   = cylinder[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = ((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq <= 1) & \
            ((coordp[2,:]-z0)>=-zl) & ((coordp[2,:]-z0)<=zl)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
        
    return p

def Sphere(sphere, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (sphere.shape[0]):
        A   = sphere[k,0]
        asq = sphere[k,1]**2
        bsq = sphere[k,1]**2
        csq = sphere[k,1]**2
        x0  = sphere[k,4]
        y0  = sphere[k,5]
        z0  = sphere[k,6]
        phi = sphere[k,7]
        theta = sphere[k,8]
        psi   = sphere[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = (((coordp[0,:]-x0)**2/asq + (coordp[1,:]-y0)**2/bsq + (coordp[2,:]-z0)**2/csq) <= 1)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
    
    return p

def Cuboid(cuboid, size=[128,128,128], spacing=[1,1,1]):
    nx = size[0]
    ny = size[1]
    nz = size[2]
    fov = [nx*spacing[0], ny*spacing[1], nz*spacing[2]]
    xr = np.arange(nx)*spacing[0] - fov[0]/2.0
    yr = np.arange(ny)*spacing[1] - fov[1]/2.0
    zr = np.arange(nz)*spacing[2] - fov[2]/2.0
    [y,x,z] = np.meshgrid(yr,xr,zr,indexing='ij')
    
    p = np.zeros((nx,ny,nz))
    coord = np.asarray([x.flatten(),y.flatten(), z.flatten()])
    p = p.flatten()   
    
    for k in range (cuboid.shape[0]):
        A  = cuboid[k,0]
        xl = cuboid[k,1]
        yl = cuboid[k,2]
        zl  = cuboid[k,3]
        x0  = cuboid[k,4]
        y0  = cuboid[k,5]
        z0  = cuboid[k,6]
        phi = cuboid[k,7]
        theta = cuboid[k,8]
        psi   = cuboid[k,9]
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
    
        # Euler rotation matrix
        alpha = np.asarray([[cpsi*cphi-ctheta*sphi*spsi,   cpsi*sphi+ctheta*cphi*spsi,  spsi*stheta],
                            [-spsi*cphi-ctheta*sphi*cpsi,  -spsi*sphi+ctheta*cphi*cpsi, cpsi*stheta],
                            [stheta*sphi,                  -stheta*cphi,                ctheta]])      
    
        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha,coord)
    
        m = ((coordp[0,:]-x0)>=-xl) & ((coordp[0,:]-x0)<=xl) & \
            ((coordp[1,:]-y0)>=-yl) & ((coordp[1,:]-y0)<=yl) & \
            ((coordp[2,:]-z0)>=-zl) & ((coordp[2,:]-z0)<=zl)
        idx = m.nonzero()
        p[idx] = p[idx] + A 
    
    p = p.reshape((nx,ny,nz))
    
    return p


def main():
    # create tmp directory if not exist
    if not os.path.exists(config['saveFolder']):
        os.mkdir(config['saveFolder'])
    
    for count in range(config['count']):
        print('count=%d' % (count))

        x = np.zeros((config['mat_size'][0], config['mat_size'][1], config['mat_size'][2]))
        mask = np.ones((config['mat_size'][0], config['mat_size'][1], config['mat_size'][2]))
           
        index = mask.nonzero()
        n = np.random.randint(len(index[0]), size=6)
        ix, iy, iz = index[0][n], index[1][n], index[2][n]
        nx, ny, nz = config['mat_size'][0], config['mat_size'][1], config['mat_size'][2]   
        xs, ys, zs = config['voxel_size'][0], config['voxel_size'][1], config['voxel_size'][2]
        
        
        for ii in range(len(n)):
            suscp_value = np.random.randn(1)[0]*0.1 + 1.5
            if np.random.randint(3, size=1)[0]==0:
                suscp_value *= -1
            
            shape = np.zeros((1,10))
            shape[0,0] = suscp_value
            shape[0,1] = np.random.randint(5, size=1)+2
            shape[0,2] = np.random.randint(5, size=1)+2
            shape[0,3] = np.random.randint(5, size=1)+2
            shape[0,4] = (ix[ii]-nx/2.0)*xs
            shape[0,5] = (iy[ii]-ny/2.0)*ys
            shape[0,6] = (iz[ii]-nz/2.0)*zs
            shape[0,7] = np.random.randint(360,size=1)[0]/180.0*np.pi
            shape[0,8] = np.random.randint(360,size=1)[0]/180.0*np.pi
            shape[0,9] = np.random.randint(360,size=1)[0]/180.0*np.pi                   
            
            idx = np.random.randint(4, size=1)[0] 
            idx = 0
            if idx == 0:
                x += Ellipsoid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
            elif idx == 1:
                x += Sphere(shape, size=[nx, ny, nz], spacing=[xs, ys, zs]) 
            elif idx == 2:
                x += Cylinder(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
            else:
                x += Cuboid(shape, size=[nx, ny, nz], spacing=[xs, ys, zs])
                
        
        mask = 1*(x!=0)

        # save data 
        savePath = os.path.join(os.path.abspath(config['saveFolder']),("%s_%d"%(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3],1)))
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        else:
            shutil.rmtree(savePath)
            os.mkdir(savePath)
        affMatrix = np.zeros((4,4))
        affMatrix[0,0] = config['voxel_size'][0]
        affMatrix[1,1] = config['voxel_size'][1]
        affMatrix[2,2] = config['voxel_size'][2]

        try:
            saveNifti(x, os.path.join(savePath, 'suscp.nii.gz'), affMatrix)
            saveNifti(mask*1, os.path.join(savePath, 'Mask.nii.gz'), affMatrix)
        except:
            continue
   
if __name__ == "__main__":
    main()


