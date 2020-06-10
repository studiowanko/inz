def plot_list_of_points_on_map(latitudes, longitudes, description):
    """Simple plotter of locations on earth from given lists of lat and lon and description.
    Parameters
    ----------
    latitudes : list
        List of satellite information. See sat_info function for more details.
    longitudes : list
        List of satellite information. See sat_info function for more details.
    description: str
        Description of showed photo.
    Returns
    -------
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    fig = plt.figure(figsize=(8, 8))
    mp = Basemap(projection='lcc', resolution=None,
                 width=35E6, height=35E6,
                 lat_0=1, lon_0=1, )
    mp.etopo(scale=0.5, alpha=0.5)
    x, y = mp(longitudes, latitudes)
    plt.plot(x, y, 'ok', markersize=4)
    fig.suptitle(description, fontsize=20, fontweight='bold')
    return ()


def plot_multiple_images(array, dpi, images):
    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    f, axarr = plt.subplots(array[0], array[1], dpi=dpi)
    k = 0
    image_datas = []
    for i in range(array[0]):
        for j in range(array[1]):
            image_datas.append(mpimg.imread(images[k]))
            axarr[i, j].imshow(image_datas[k])
            k = k + 1
    return ()


def draw_poins_on_map(ext_lats, ext_lons, res_path, dpi):
    # https://stackoverflow.com/questions/44488167/plotting-lat-long-points-using-basemap
    import matplotlib.pyplot as plt
    import os
    os.environ['PROJ_LIB'] = '/home/jan/anaconda3/envs/inz2/lib/python3.7/site-packages/mpl_toolkits/basemap'
    from mpl_toolkits.basemap import Basemap

    fig = plt.figure(figsize=(8, 8))
    # determine range to print based on min, max lat and lon of the data
    margin = 5  # buffer to add to the range
    lat_min = min(ext_lats) - margin
    lat_max = max(ext_lats) + margin
    lon_min = min(ext_lons) - margin
    lon_max = max(ext_lons) + margin

    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lat_0=(lat_max - lat_min) / 2,
                lon_0=(lon_max - lon_min) / 2,
                projection='merc',
                resolution='l',
                area_thresh=10000.,
                )

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='white', lake_color='#46bcec')
    # convert lat and lon to map projection coordinates
    lons, lats = m(ext_lons, ext_lats)
    # plot points as red dots
    m.scatter(lons, lats, marker='o', color='r', zorder=5)
    plt.savefig(res_path, dpi=dpi)
    return ()


def draw_polygons_on_map(polygons, lines, points, composite, files, photo_extent , result_path, projection='Stereographic', dpi=200):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from satpy.scene import Scene

    col = ['g', 'b', 'y', 'm', 'k', 'c', 'w']
    col2 = ['m', 'k', 'c', 'w']
    col3 = ['y', 'y', 'y']
    scn = Scene(filenames=files)
    scn.load([composite])
    new_scn = scn
    crs = new_scn[composite].attrs['area'].to_cartopy_crs()
    proj = getattr(ccrs,projection)()
    ax0 = plt.axes(projection=proj)
    if photo_extent == 'global':
        ax0.set_global()
    else:
        ax0.set_extent(photo_extent, crs=ccrs.PlateCarree())
    ax0.gridlines()
    ax0.coastlines(color='r')
    plt.plot()

    # Plot 1 polygon:

    for i, polygon in enumerate(polygons):
        poly_lats, poly_lons = polygon[0], polygon[1]
        for la, lo in zip(poly_lats, poly_lons):
            plt.fill(lo, la, transform=ccrs.PlateCarree(), color=col[i])

    # Plot line:
    for i, line in enumerate(lines):
        poly_lats, poly_lons = line[0], line[1]
        for la, lo in zip(poly_lats, poly_lons):
            plt.plot(lo, la, 'ok', markersize=4,transform=ccrs.PlateCarree(), color=col2[i])

    # Plot points:
    for i, point in enumerate(points):
        la, lo = point[0], point[1]
        print(la,lo)
        plt.plot(lo, la, 'ok', markersize=4, transform=ccrs.PlateCarree(), color=col3[i])

    plt.imshow(new_scn[composite], transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    plt.savefig(result_path, dpi=dpi)
    return ()

def plot_multi_list_of_points_on_map(lats_arr, lons_arr, description):
    """Simple plotter of locations on earth from given lists of lat and lon and description.
    Parameters
    ----------
    latitudes : list
        List of satellite information. See sat_info function for more details.
    longitudes : list
        List of satellite information. See sat_info function for more details.
    description: str
        Description of showed photo.
    Returns
    -------
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    col = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'w']
    if len(lats_arr) > 8:
        raise ValueError('Not enough colours defined!')

    fig = plt.figure(figsize=(8, 8))
    mp = Basemap(projection='lcc', resolution=None,
                 width=35E6, height=35E6,
                 lat_0=1, lon_0=1, )
    mp.etopo(scale=0.5, alpha=0.5)
    for i, (lat, lon) in enumerate(zip(lats_arr, lons_arr)):
        x, y = mp(lon, lat)
        plt.plot(x, y, 'ok', markersize=4, color=col[i])

    fig.suptitle(description, fontsize=20, fontweight='bold')
    return ()


def plot_coastlines_on_map(composite, files, photo_extent, points, result_path, dpi=800):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from satpy.scene import Scene
    # fig = plt.figure(figsize=(16,12))
    # col = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'w']

    scn = Scene(filenames=files)
    scn.load([composite])

    new_scn = scn
    crs = new_scn[composite].attrs['area'].to_cartopy_crs()
    ax1 = plt.axes(projection=ccrs.Mercator())
    ax1.set_extent(photo_extent)
    #ax1.drawcountries()
    #ax1.drawstates()
    #ax1.gridlines()
    # ax.coastlines(resolution='50m', color='red')
    ax1.coastlines(color='r')
    plt.plot()
    # ax.gridlines()
    # ax.set_global()
    for i, (lat, lon) in enumerate(zip(points[0], points[1])):
        plt.plot(lon, lat, 'r*', ms=15, transform=ccrs.Geodetic())
    # plt.plot(400, 2000, 'ok', markersize=400, color=col[i],projection=crs)

    #   fig.suptitle(description, fontsize=20, fontweight='bold')
    # ax.scatter(10,40,latlon=True,color='blue')
    plt.imshow(new_scn['VIS006'], transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    # cbar = plt.colorbar()
    # cbar.set_label("Kelvin")
    plt.savefig(result_path, dpi=dpi)
    # plt.show()
    # plt.imsave(result_path, dpi=dpi)
    return ()


def save_collage(images, img, PATH, images_per_row=5, frame_width=1920):
    import PIL, os, glob
    from PIL import Image
    from math import ceil, floor
    # frame_width = 1920
    padding = 2

    os.chdir(PATH)
    # get the first 30 images

    img_width, img_height = Image.open(images[0]).size
    sf = (frame_width - (images_per_row - 1) * padding) / (images_per_row * img_width)  # scaling factor
    scaled_img_width = ceil(img_width * sf)  # s
    scaled_img_height = ceil(img_height * sf)

    number_of_rows = ceil(len(images) / images_per_row)
    frame_height = ceil(sf * img_height * number_of_rows)

    new_im = Image.new('RGB', (frame_width, frame_height))

    i, j = 0, 0
    for num, im in enumerate(images):
        if num % images_per_row == 0:
            i = 0
        im = Image.open(im)
        # Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width, scaled_img_height))
        # Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j // images_per_row) * scaled_img_height
        new_im.paste(im, (i, y_cord))
        #print(i, y_cord)
        i = (i + scaled_img_width) + padding
        j += 1

    new_im.save(img, "PNG", quality=95, optimize=True, progressive=True)
    return ()
