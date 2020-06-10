def img_projection(img_fname,source_proj,target_proj,target_res,res_fname):
	from cartopy.img_transform import warp_array
	import numpy as np    
	import PIL.Image
	
	img = PIL.Image.open(img_fname)
	result_array, extent = warp_array(np.array(img),source_proj,target_proj,target_res)
	result = PIL.Image.fromarray(result_array)
	result.save(res_fname)

	return()
	
def img_proj_mill(fname,res_name,resolution,height):
	import cartopy.crs as ccrs
	from cartopy.img_transform import warp_array
	import numpy as np    
	import PIL.Image

	img = PIL.Image.open(fname)

	result_array, extent = warp_array(np.array(img),
									  source_proj=ccrs.Geostationary(satellite_height=height),
									  target_proj=ccrs.Miller(),
									  target_res=resolution)

	result = PIL.Image.fromarray(result_array)
	result.save(res_name)
	return()
	
def img_proj_plate_carree(fname,res_name,resolution,height):
	import cartopy.crs as ccrs
	from cartopy.img_transform import warp_array
	import numpy as np    
	import PIL.Image

	img = PIL.Image.open(fname)

	result_array, extent = warp_array(np.array(img),
									  source_proj=ccrs.Geostationary(satellite_height=height),
									  target_proj=ccrs.PlateCarree(),
									  target_res=resolution)

	result = PIL.Image.fromarray(result_array)
	result.save(res_name)
	return()
	
	

def img_proj_merc(fname,res_name,resolution,height):
	import cartopy.crs as ccrs
	from cartopy.img_transform import warp_array
	import numpy as np    
	import PIL.Image

	img = PIL.Image.open(fname)

	result_array, extent = warp_array(np.array(img),
									  source_proj=ccrs.Geostationary(satellite_height=height),
									  target_proj=ccrs.Mercator(),
									  target_res=resolution)

	result = PIL.Image.fromarray(result_array)
	result.save(res_name)
	return()