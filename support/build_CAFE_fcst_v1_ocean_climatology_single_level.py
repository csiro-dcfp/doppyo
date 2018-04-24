import pandas as pd
import xarray as xr
import numpy as np
from pylatte import utils
from ipywidgets import FloatProgress

# Location of forecast data -----
fcst_folder = '/OSM/CBR/OA_DCFP/data/model_output/CAFE/forecasts/v1/'
fcst_filename = 'ocean_daily*'

# fields = pd.DataFrame( \
#         {'name_CAFE': ['sst', 'patm_t', 'eta_t', 'sss', 'u_surf', 'v_surf'],
#          'name_std' : ['sst', 'patm',   'eta',   'sss', 'u_s',    'v_s']}
                     )
fields = pd.DataFrame( \
        {'name_CAFE': ['sst'],
         'name_std' : ['sst']}
                      
name_dict = fields.set_index('name_CAFE').to_dict()['name_std']

# Initial dates to include (takes approximately 1 min 30 sec per date) -----
init_dates = pd.date_range('2002-2','2016-5' , freq='1MS')

# Ensembles to include -----
ensembles = range(1,12)

path = fcst_folder + '/yr2016/mn1/OUTPUT.1/' + fcst_filename + '.nc'
dataset = xr.open_mfdataset(path, autoclose=True)
time_use = dataset.time[:366]

years = range(2002,2017)
months = range(1,13)
ensembles = range(1,12)

for year in years:
    print(year)
    print('----------')
    for idx, variable in enumerate(fields['name_CAFE']):
        print(variable)
        
        savename = 'cafe.fcst.v1.ocean.' + fields['name_std'][idx] + '.' + str(year) + '.clim.nc'
        try:
            temp = xr.open_mfdataset('/OSM/CBR/OA_DCFP/data/intermediate_products/pylatte_climatologies/' + savename, autoclose=True)
            print('    Already exists')
        except:
            
            fcst_list = []
            for month in months:

                ens_list = []
                ens = []
                empty = True
                for ie, ensemble in enumerate(ensembles):

                    path = fcst_folder + '/yr' + str(year) + '/mn' + str(month) + \
                           '/OUTPUT.' + str(ensemble) + '/' + fcst_filename + '.nc'

                    # Try to stack ensembles into a list -----
                    try:
                        dataset = xr.open_mfdataset(path, autoclose=True)[variable]
                        if 'xt_ocean' in dataset.dims:
                            dataset = dataset.rename({'xt_ocean':'lon_t','yt_ocean':'lat_t'})
                        if 'xu_ocean' in dataset.dims:
                            dataset = dataset.rename({'xu_ocean':'lon_u','yu_ocean':'lat_u'})
                        ens_list.append(dataset.rename(fields['name_std'][idx]))
                        ens.append(ie+1)
                        empty = False
                    except OSError:
                        # File does not exist -----
                        pass

                # Concatenate ensembles -----
                if empty == False:
                    ens_object = xr.concat(ens_list, dim='ensemble')
                    ens_object['ensemble'] = ens

                    # Stack concatenated ensembles into a list for each month in a year -----                       
                    fcst_list.append(ens_object)

            # Concatenate all months within year -----
            ds = xr.concat(fcst_list, dim='time')

            # Rechunk for chunksizes of at least 1,000,000 elements -----
            ds = utils.prune(ds.chunk(chunks={'ensemble' : len(ds.ensemble), 
                                              'time' : len(ds.time)}).squeeze())

            # Make month_day array of month-day -----
            m = np.array([str(i).zfill(2) + '-' for i in ds.time.dt.month.values])
            d = np.array([str(i).zfill(2)  for i in ds.time.dt.day.values])
            md = np.core.defchararray.add(m, d)

            # Replace time array with month_day array and groupby -----
            ds['time'] = md
            ds_clim = ds.groupby('time').mean(dim=['time','ensemble'],keep_attrs=True)

            # Fill time with presaved time -----
            ds_clim['time'] = time_use
            ds_clim.time.attrs['long_name'] = 'time'
            ds_clim.time.attrs['cartesian_axis'] = 'T'
            ds_clim.time.attrs['calendar_type'] = 'JULIAN'
            ds_clim.time.attrs['bounds'] = 'time_bounds'

            # Save and delete -----
            with utils.timer():
                ds_clim.to_netcdf(path='/OSM/CBR/OA_DCFP/data/intermediate_products/pylatte_climatologies/' + savename,
                                  mode = 'w',
                                  encoding = {'time':{'dtype':'float','calendar':'JULIAN',
                                                      'units':'days since 0001-01-01 00:00:00'}}) 

        # if 'ds' in locals():
        #     del ds, ds_clim