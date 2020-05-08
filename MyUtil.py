' Purpose of this script '
' 25Sep2019 '
__author__ = 'YeLiu'

import numpy as np
import pandas as pd
import xarray as xr
#import xesmf as xe
from collections import OrderedDict
import seaborn as sns
from scipy import stats

def convert_time(xtime=(195001,201512)):
    return pd.to_datetime(xtime, format='%Y%m', errors='ignore').strftime("%Y-%m-%d")

class MyDataControl(object):
    def read_netcdf(self, fname, vname, opt=None, **kwargs):
        if kwargs.get('debug', False):
           print(fname, vname)
        try:
           ds = xr.open_dataset(fname, **kwargs)
        except:
           ds = xr.open_dataset(fname, decode_times=False, **kwargs)
           ds.coords['time'].attrs['calendar'] = '360_day'
           ds.coords['time'] = xr.decode_cf(ds, use_cftime=True).time
           # convert cftime to datetime64
           ds.coords['time'] = np.array([np.datetime64(ii) for ii in ds.coords['time'].data])


        if vname is not 'ALLVAR': ds    = ds[vname]
        for idim in ds.dims:
            if idim in ('lon', 'longitude', 'lon_126', 'g0_lon_1'): ds = ds.rename({idim: 'lon'})
            if idim in ('lat', 'latitude',  'lat_126', 'g0_lat_0'): ds = ds.rename({idim: 'lat'})
            if idim in ('lv_ISBL3'):  ds = ds.rename({idim: 'level'})

        # convert float64 to float32
        if kwargs.get('convert_64_to_32', True):
            if isinstance(ds, xr.DataArray):
                if ds.dtype == np.float64:
                    ds = ds.astype(np.float32)
            if isinstance(ds, xr.Dataset):
                for v in ds:
                    if ds[v].dtype == np.float64:
                        ds[v] = ds[v].astype(np.float32)
            for idim in ds.dims:
                if ds.coords[idim].dtype == np.float64:
                    ds.coords[idim] = ds.coords[idim].astype(np.float32)

        if opt is not None:
           for iopt in opt:
              ds = getattr(self, iopt)(ds)
        return ds
 
    def lon_flip(self, da, lon_coord='lon'):
        lon   = da[lon_coord]
        if len(lon) == 720:
            shift = int(len(lon)/2) 
        else:
            shift = int(len(lon)/2) - 1
        da = da.roll(lon=shift, roll_coords=True)
        da.attrs['lon_flip'] = True
        if max(lon) > 180:
            return da.assign_coords({lon_coord: (((da[lon_coord] + 180) % 360)-180)})
        else:
            return da.assign_coords({lon_coord: (((da[lon_coord] + 180) % 360))})

    def lat_reverse(self, da, lat_coord='lat'):
        da.attrs['lat_reverse'] = "True"
        return da.reindex(**{lat_coord:da[lat_coord][::-1]})

    def topo_mask(self, da, **kwargs):
        vftopo = kwargs.get('vftopo', ())
        vrange = kwargs.get('vrange', [0, 1])
        topo   = self.read_netcdf(*vftopo).squeeze()

        if kwargs.get('remap', False):
            topo = topo.interp(lat=da.lat, lon=da.lon) 

        da.coords['topo'] = (('lat', 'lon'), topo)
        da = da.where((da.topo>=vrange[1])|(da.topo<vrange[0]))
        if not kwargs.get('keepmask', False): del da['topo']
        return da


MDC = MyDataControl()

class monthstring(object):
    monthmap = (('J', 'Jan', 'January',   1),
                ('F', 'Feb', 'February',  2),
                ('M', 'Mar', 'March',     3),
                ('A', 'Apr', 'April',     4),
                ('M', 'May', 'May',       5),
                ('J', 'Jun', 'June',      6),
                ('J', 'Jul', 'July',      7),
                ('A', 'Aug', 'August',    8),
                ('S', 'Sep', 'September', 9),
                ('O', 'Oct', 'October',  10),
                ('N', 'Nov', 'November', 11),
                ('D', 'Dec', 'December', 12),
               )

    def one_letter_abbr_identify(self, s, s1=None):
        maps = [v[0] for v in monthmap]
        self.monthid = maps.index(s)+1 if s not in ('J', 'M', 'A') else None
        if s == 'J':
            if s1 == 'F': self.monthid = 1
            if s1 == 'J': self.monthid = 6
            if s1 == 'A': self.monthid = 7
        if s == 'M':
            if s1 == 'A': self.monthid = 3
            if s1 == 'J': self.monthid = 5
        if s == 'A':
            if s1 == 'M': self.monthid = 4
            if s1 == 'S': self.monthid = 8
   
    def one_letter_abbr(self, monthid):
        out = [ monthmap[i-1][0] for i in monthid]
        out = ''.join(out)
        return out

    def three_letter_abbr(self, monthid):
        out = [ monthmap[i-1][1] for i in monthid]
        out = '-'.join(out)
        return out

    def full_month(self, monthid):
        out = [ monthmap[i-1][2] for i in monthid]
        out = '-'.join(out)
        return out
        
class MyDataOperator(object):
    def selmonths(self, da, monstr='JJA', **kwargs):
        for i, s in monstr:
            if i == len(monstr)-1: 
                m, mmm, month, num = abbr_to_month(s)
            else:
                sn = monstr[i+1]
                m, mmm, month, num = abbr_to_month(s, sn)
             
        
    def wgt_area_ave(self, da, lat_dim='lat', lon_dim='lon', **kwargs):
        wda, clat  = self.gen_weight(da, lat_dim)
        wave = wda.mean(dim=[lat_dim,lon_dim]) / clat.mean()
        wave.name = da.name
#        attrs = wave.attrs
#        for attr, v in da.attrs.items():
#            attrs[attr] = v
#        dims   = [k for k in da.dims if k not in (lat_dim, lon_dim)]
#        coords = {k: v for k, v in da.coords.items() if k not in (lat_dim, lon_dim)}
#        wave = xr.DataArray(wave, dims=dims, coords=coords)
        return wave

    def two_sample_diff(self, xda1, xda2, dim='time', **kwargs):
        from scipy.stats import ttest_ind
        if isinstance(xda1, xr.Dataset): xda1 = xda1.to_array()
        if isinstance(xda2, xr.Dataset): xda2 = xda2.to_array()
#        da1, da2 = xr.broadcast(xda1, xda2)
        da1, da2 = (xda1, xda2)
        dim_idx = da1.dims.index(dim)
        t, p = ttest_ind(da1.data, da2.data, axis=dim_idx)
        dims   = [k for k in da1.dims if k != dim]
        coords = {k: v for k, v in da1.coords.items() if k != dim}
        p    = xr.DataArray(p, dims=dims, coords=coords)
        diff = da1.mean(dim) - da2.mean(dim) 
        return diff, p

    def season_average(self, da, season):
        mon_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Sep', 'Oct', 'Nov', 'Dec']
        if season in ('DJF', 'JJA', 'MAM', 'SON'):
            xx   = {'DJF':2,  'MAM':5, 'JJA':8, 'SON':11}
            mda  = da.rolling(time=3, min_periods=2).mean().sel(time=da['time.month']==xx[season])
#            mda  = mda.assign_coords({'time': mda['time.year']})

        if season == 'JJAS':
            mda  = da.sel(time=((da['time.month']==6 | da['time.month']==7 | da['time.month']==8 |da['time.month']==9))).\
                          groupby('time.year').mean(dim='time').rename({'year':'time'})

        if season == 'ann':
            mda = da.groupby('time.year').mean(dim='time').rename({'year':'time'})

        if season in mon_str:
            mda = da.sel(time=(da['time.month']==mon_str.index(season)+1))#.groupby('time.year').mean('time').rename({'year':'time'})

        mda.name = da.name
        return mda

    def spatial_correlation(self, xda1, xda2, lat_dim='lat', lon_dim='lon', **kwargs):
        if isinstance(xda1, xr.Dataset): xda1 = xda1.to_array()
        if isinstance(xda2, xr.Dataset): xda2 = xda2.to_array()
        da1, da2 = xr.broadcast(xda1, xda2)
        da1 = xr.where(da1.notnull() & da2.notnull(), da1, np.nan)
        da2 = xr.where(da1.notnull() & da2.notnull(), da2, np.nan)
        da1 = da1.stack(z=(lat_dim, lon_dim))#.dropna(dim='z', how='all')
        da2 = da2.stack(z=(lat_dim, lon_dim))#.dropna(dim='z', how='all')
        return xr.apply_ufunc(
            self._sci_pearsonr, da1, da2,
            input_core_dims=[['z'], ['z']],
            vectorize=True,# !Important!
            output_core_dims=[[],[]],
            )
        
    def gen_weight(self, da, lat_dim='lat', **kwargs):
        lat = da.coords[lat_dim]
        clat = xr.ufuncs.cos(xr.ufuncs.deg2rad(da.coords[lat_dim]))
        return  da * clat, clat

    def landsea_mask(self, da, land_index=1, **kwargs):
        vfmask = kwargs.get('vfmask', ('/Volumes/YeLiu/pySrc/cfs_sfc_landsea.nc', 'lsdata', ('lon_flip', 'lat_reverse')))
        ###..need edit
        mask  = MDC.read_netcdf(*vfmask).interp(lat=da.lat, lon=da.lon)
        da.coords['mask'] = (('lat', 'lon'), mask)

        land  = da.where(da.mask>=0.9)
        ocean = da.where(da.mask<0.9)

        out   = xr.concat([da, land, ocean], 'land_sea')
        out.assign_coords({'land_sea': np.array([0,1,2])})
     
#        return xr.concat([da, land, ocean], pd.Index(np.array([0, 1, 2]), name='land_sea'))
        return out
       

    def sci_linregress(self, x, y, dim='time'):
        print(x.coords)
        print(y.coords)
        return xr.apply_ufunc(self._sci_linregress, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,# !Important!
            output_core_dims=[[],[],[],[],[]],
            )

    def _sci_linregress(self, xx, yy):
        x = xx[~np.isnan(xx) & ~np.isnan(yy)]
        y = yy[~np.isnan(xx) & ~np.isnan(yy)]
        if len(x)<4 or len(y)<4:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return stats.linregress(x, y)
              

        
      
    def sci_pearsonr(self, x, y, dim='time'):
        if isinstance(x, xr.Dataset): x = x.to_array()
        if isinstance(y, xr.Dataset): y = y.to_array()
        x, y = xr.broadcast(x, y)
        xx = xr.where(y.notnull() & x.notnull(), x, np.nan).dropna(dim=dim, how='all')
        yy = xr.where(x.notnull() & y.notnull(), y, np.nan).dropna(dim=dim, how='all')
        return xr.apply_ufunc(
            self._sci_pearsonr, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,# !Important!
            output_core_dims=[[],[]],
            )

    def _sci_pearsonr(self, xx, yy):
        x = xx[~np.isnan(xx) & ~np.isnan(yy)]
        y = yy[~np.isnan(xx) & ~np.isnan(yy)]
        if len(x)<4 or len(y)<4:
            return np.nan, np.nan
        else:
            return stats.pearsonr(x, y)

    def _sci_wraper(self, func_name, *args, **kwargs):
        pass




MDO = MyDataOperator()
        

    
def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """
    Given a function `func1d(A, B, C, ..., *args, **kwargs)`  that acts on 
    multiple one dimensional arrays, apply that function to the N-dimensional
    arrays listed by `arrs` along axis `axis`
    
    If `arrs` are one dimensional this is equivalent to::
    
        func1d(*arrs, *args, **kwargs)
    
    If there is only one array in `arrs` this is equivalent to::
    
        numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)
        
    All arrays in `arrs` must have compatible dimensions to be able to run
    `numpy.concatenate(arrs, axis)`
    
    Arguments:
        func1d:   Function that operates on `len(arrs)` 1 dimensional arrays,
                  with signature `f(*arrs, *args, **kwargs)`
        axis:     Axis of all `arrs` to apply the function along
        arrs:     Iterable of numpy arrays
        *args:    Passed to func1d after array arguments
        **kwargs: Passed to func1d as keyword arguments
    """
    # Concatenate the input arrays along the calculation axis to make one big
    # array that can be passed in to `apply_along_axis`
    carrs = np.concatenate(arrs, axis)
    
    # We'll need to split the concatenated arrays up before we apply `func1d`,
    # here's the offsets to split them back into the originals
    offsets=[]
    start=0
    for i in range(len(arrs)-1):
        start += arrs[i].shape[axis]
        offsets.append(start)
            
    # The helper closure splits up the concatenated array back into the components of `arrs`
    # and then runs `func1d` on them
    def helperfunc(a, *args, **kwargs):
        arrs = np.split(a, offsets)
        return func1d(*[*arrs, *args], **kwargs)
    
    # Run `apply_along_axis` along the concatenated array
    return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)

def plot_axes(ax, fig=None, geometry=(1,1,1)):
    '''
    reuse axes generated by previous plot
    '''
    if fig is None: fig = plt.figure()
    ax.change_geometry(*geometry)
    fig._axstack.add(fig._make_key(ax), ax)
    return fig

def snsBlendColor(cList, n_color=21, as_cmap=True, **kw):
    sns.set()
    cmap = sns.blend_palette(cList, n_color, as_cmap=as_cmap, **kw )
#    sns.palplot(sns.muted_palette(cList, n_color))
    return cmap

def printVarSummary(array):
    if isinstance(array, xr.core.dataarray.DataArray):
        print('----')
        print(array.name)
        print(type(array))
        print(array.shape); print(array.dims); print(array.coords)


from timeit import default_timer as timer
import sys
def timefunc(func, *args, **kwargs):
    """Time a function. 

    args:
        iterations=3

    Usage example:
        timeit(myfunc, 1, b=2)
    """
    try:
        iterations = kwargs.pop('iterations')
    except KeyError:
        iterations = 3
    elapsed = sys.maxsize
    for _ in range(iterations):
        start = timer()
        result = func(*args, **kwargs)
        elapsed = min(timer() - start, elapsed)
    print(('Best of {} {}(): {:.9f}'.format(iterations, func.__name__, elapsed)))
    return result

############################################ TEST ##############################################
def _test_read_netcdf():
#     da = MDC.read_netcdf('example1.nc', 'air')
     da = MDC.read_netcdf('/Volumes/YeLiu/CFS_data/TMP/TMP_SSiB1_10th_mon.nc', ['tmp'])
#     da = MDC.read_netcdf('/Volumes/YeLiu/cfs_anl/observation/U850_NCEP_mon.nc', ['uwnd'])
#     print(xr.decode_cf(da, decode_times=False, use_cftime=True).coords['time'])
#     print(type(da.coords['time'].data[0]))
     print(da)
    
if __name__ is '__main__':
#import pyLib.util_ye as myUtil
#    import CCFunc as CCF
#    import timeit
#    
#    ds  = myReadNetCDF(CCF.DIROBS+'dswrf_TP_1979-2015_0.5d.day.nc',   ['srad'])
#    da  = ds.srad.chunk({'lat':25, 'lon':25})
#    timefunc(myWgtAreaAve, da)
#    timefunc(mySFCTopo, da)
#    printVarSummary(da)
    _test_read_netcdf()


#    snsBlendColor(['white','blue','green','yellow','orange','red'])

