def show_simple_position(stations_url, sat_name, times, description):
    # TO DO: FIND METHOD FOR HANDLING sexagesimal <=> decimal units conversion
    from skyfield.api import Topos, load
    # %matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    satellites = load.tle(stations_url, reload=True)
    satellite = satellites[sat_name]

    t = times

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='lcc', resolution=None,
                width=35E6, height=35E6,
                lat_0=1, lon_0=1, )
    m.etopo(scale=0.5, alpha=0.5)

    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()
    sign1, d1, m1, s1 = subpoint.latitude.signed_dms()
    lat = sign1 * (d1 + m1 / 60 + s1 / 3600)
    sign2, d2, m2, s2 = subpoint.longitude.signed_dms()
    lon = sign2 * (d2 + m2 / 60 + s2 / 3600)

    x, y = m(lon, lat)
    plt.plot(x, y, 'ok', markersize=4)
    fig.suptitle(description, fontsize=20, fontweight='bold')
    return ()


def satellite_info(earth_radius, satellite_height, camera_hfov, camera_vfov):
    """Prints information about satellite earth coverage

    Parameters
    ----------
    earth_radius : double
        Local earth radius
    satellite_height : double
        Shortest distance from satellite to earth.
    camera_hfov : double
        Horizontal field of view of camera on satellite in radians.
    camera_vfov : double
        Vertical field of view of camera on satellite in radians.

    Returns
    -------
    tangent_point: double
        Distance of line tangent to earth from satellite
    vis_arc: double
        Length of arc from visible bowl of earth
    vis_diameter: double
        Diameter of circle projected from visible bowl of earth
    vis_percentage: double
        Percentage of visable earth area max 50% in  infinity
    vis_angle: double
        Visible angle between tangent points - in degrees
        :param camera_vfov:
        :param camera_hfov:
        :param earth_radius:
        :param satellite_height:
    """

    import math as m
    from wutsat.fun import mat_fun

    y_satellite = earth_radius + satellite_height
    a = m.sqrt((m.pow(y_satellite, 2) - m.pow(earth_radius, 2)) / m.pow(earth_radius, 2))
    x2 = (a * m.pow(earth_radius, 2)) / (y_satellite)
    y2 = -m.pow(a, 2) * m.pow(earth_radius, 2) / y_satellite + y_satellite
    # what is what:
    tangent_point = m.sqrt(m.pow(x2, 2) + m.pow((y2 - y_satellite), 2))
    gamma1 = (m.pi / 2 - m.atan(y2 / x2))
    vis_arc = 2 * earth_radius * gamma1
    beta1 = m.pi / 2 - gamma1
    vis_diameter = 2 * x2
    vis_percentage = vis_arc / (2 * m.pi * earth_radius) * 100
    vis_angle = 2 * m.atan(y2 / x2) * 180 / m.pi
    # https://www.edmundoptics.com/resources/application-notes/imaging/understanding-focal-length-and-field-of-view/
    camera_dfov = m.sqrt(m.pow(camera_hfov, 2) + (m.pow(camera_vfov, 2)))
    # TO DO: Raise warning if camera dfov is greater
    if camera_dfov / 2 > beta1:
        print("Camera captures the whole earth!")
        return None,None,None,None,None,None,None,None,None,None,None
        # raise ValueError('Camera captures the whole earth!')
    gamma2 = mat_fun.find_angle_analytic(earth_radius, satellite_height, camera_dfov / 2)
    camera_visible_arc = 2 * earth_radius * gamma2
    camera_to_visability = (1 - m.cos(gamma2)) / (1 - m.cos(gamma1)) * 100
    camera_to_earth = (1 - m.cos(gamma2)) / 2 * 100
    percentage_arc_to_arc = camera_visible_arc / vis_arc * 100
    return tangent_point, vis_arc, vis_diameter, vis_percentage, vis_angle, camera_visible_arc, camera_to_visability, camera_to_earth, percentage_arc_to_arc, gamma2


def satellite_basic(earth_radius, satellite_height):
    """Prints information about satellite earth coverage

    Parameters
    ----------
    earth_radius : double
        Local earth radius
    satellite_height : double
        Shortest distance from satellite to earth.
    camera_hfov : double
        Horizontal field of view of camera on satellite.
    camera_vfov : double
        Vertical field of view of camera on satellite.

    Returns
    -------
    tangent_point: double
        Distance of line tangent to earth from satellite
    vis_arc: double
        Length of arc from visible bowl of earth
    vis_diameter: double
        Diameter of circle projected from visible bowl of earth
    vis_percentage: double
        Percentage of visable earth area max 50% in  infinity
    vis_angle: double
        Visible angle between tangent points - in degrees
        :param camera_vfov:
        :param camera_hfov:
        :param earth_radius:
        :param satellite_height:
    """

    import math as m
    y_satellite = earth_radius + satellite_height
    gamma = m.acos(earth_radius / y_satellite)
    # what is what:
    tangent_point = m.sqrt(m.pow(y_satellite, 2) - m.pow(earth_radius, 2))
    vis_arc = 2 * gamma * earth_radius
    vis_diameter = 2 * m.sin(gamma) * earth_radius
    vis_area = 2 * m.pi * m.pow(earth_radius, 2) * (1 - m.cos(gamma))
    vis_percentage = 0.5 * (1 - m.cos(gamma)) * 100

    return tangent_point, vis_arc, vis_diameter, vis_percentage, vis_area, gamma


def find_point(lat, lon, direction, distance, radius=6371000):
    """Finds point away from given position along great circle
    Source math: https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lat : double
        Latitude of starting point in degrees
    lon : double
        Longitude of starting point in degrees
    direction : double
        Angle in radians calculating from north clockwise
    distance : double
        Distance to travel in meters
    Returns
    -------
    latitude: double
        Latitude of ending point in decimals
    longitude: double
        Longitude of ending point in decimals
    """

    import math as m
    lat = lat * m.pi / 180
    lon = lon * m.pi / 180
    delta = distance / radius
    latitude = m.asin(m.sin(lat) * m.cos(delta) + m.cos(lat) * m.sin(delta) * m.cos(direction))
    longitude = lon + m.atan2(m.sin(direction) * m.sin(delta) * m.cos(lat), m.cos(delta) - m.sin(lat) * m.sin(latitude))

    longitude = (longitude * 180 / m.pi + 540) % 360 - 180
    latitude = latitude * 180 / m.pi

    return latitude, longitude


def find_distance_along_great_circle(lat1, lon1, lat2, lon2, radius=6371000):
    """"Finds distance between points along great circle
    Source math: https://www.movable-type.co.uk/scripts/latlong.html
    Parameters
    TODO: add description to this fun
    ----------
    """
    from math import radians as to_rad
    from math import sin, cos, atan2, sqrt

    fi1 = to_rad(lat1)
    fi2 = to_rad(lat2)
    delta_fi = to_rad(lat2 - lat1)
    delta_lambda = to_rad(lon2 - lon1)

    a = sin(delta_fi / 2) * sin(delta_fi / 2) + cos(fi1) * cos(fi2) * sin(delta_lambda / 2) * sin(delta_lambda / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = radius * c
    return distance


def square_sat(lat, lon, distance):
    """Returns four coordinates x0,x1,y0,y1 of square surrounding given point with a distance

    Parameters
    ----------
    lat : double
        Latitude of starting point
    lon : double
        Longitude of starting point
    distance : double
        Distance to trave in meters
    Returns
    -------
    lat0: double
        Latitude upper from center
    lat1: double
        Latitude lower from center
    lon0: double
        Longitude on left from center
    lon1: double
        Longitude on right from center
    """
    import math as m

    lat1, lonx = find_point(lat, lon, 0, distance)
    lat0, lonx = find_point(lat, lon, m.pi, distance)
    latx, lon1 = find_point(lat, lon, m.pi / 2, distance)
    latx, lon0 = find_point(lat, lon, m.pi * 3 / 2, distance)

    return lat0, lat1, lon0, lon1


def square_sat_list(lat, lon, distance):
    """Returns four coordinates x0,x1,y0,y1 of square surrounding given list of points with a distances

    Parameters
    ----------
    lats : list of double
        Latitude of starting point
    lon : list of double
        Longitude of starting point
    distance : list of double
        Distance to travel in meters
    Returns
    -------
    lat0: double
        Latitude upper from center
    lat1: double
        Latitude lower from center
    lon0: double
        Longitude on left from center
    lon1: double
        Longitude on right from center
    """
    import math as m

    lat1, lat0, lon1, lon0 = ([],) * 4
    length = range(len(lat))
    lat1 = [find_point(lat[i], lon[i], 0, distance[i])[0] for i in length]
    lat0 = [find_point(lat[i], lon[i], m.pi, distance[i])[0] for i in length]
    lon1 = [find_point(lat[i], lon[i], m.pi / 2, distance[i])[1] for i in length]
    lon0 = [find_point(lat[i], lon[i], m.pi * 3 / 2, distance[i])[1] for i in length]
    # Finding max and min should return extent of points
    la0 = min(lat1 + lat0)
    la1 = max(lat1 + lat0)
    lo0 = min(lon1 + lon0)
    lo1 = max(lon1 + lon0)

    return lo0, lo1, la0, la1,


def polygon_surrounding_sat(latitude, longitude, distance, div_number):
    """Returns four coordinates x0,x1,y0,y1 of square surrounding given area

    Parameters
    ----------
    latitude : double
        Latitude of starting point
    longitude : double
        Longitude of starting point
    distance : double
        Distance to trave in meters
    div_number: int
        Number of points returned.
    Returns
    -------
    lat: double
        Latitude upper from center
    lon: double
        Longitude on left from center
    """
    import math as m
    import numpy as np

    lat = np.zeros(div_number, dtype=float)
    lon = np.zeros(div_number, dtype=float)
    lat = []
    lon = []

    div = np.arange(1, div_number + 1, dtype=float) * 2 * m.pi / div_number

    for directions in div:
        x, y = find_point(latitude, longitude, directions, distance)
        lat = np.append(lat, x)
        lon = np.append(lon, y)

    return lat, lon


def get_sat_positions(time, path_to_tle, sat_name):
    """Returns arrays of lat,lon of satellite in given sets of 'time'
    time, delta_time, num_of_div, tle_path
    Parameters
    ----------
    time : list
        List of times in skyfield.timescale format.
    path_to_tle : string
        Path to TLE file
    sat_name: string
        Name of satellite to be looked for.
    Returns
    -------
    lat: array
        Latitudes in array
    lon: array
        Longitude in array
    alt: array
        Height of satellite in meters.
    rad:
        Local earth radius in meters. TODO: right now constant radius
    """

    from skyfield.api import load
    import numpy as np

    satellites = load.tle(path_to_tle, reload=True)
    satellite = satellites[sat_name]

    t = time

    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()
    sign1, d1, m1, s1 = subpoint.latitude.signed_dms()
    lat = sign1 * (d1 + m1 / 60 + s1 / 3600)
    sign2, d2, m2, s2 = subpoint.longitude.signed_dms()
    lon = sign2 * (d2 + m2 / 60 + s2 / 3600)

    alt = subpoint.elevation.m
    # TODO: Find the way to get local earth radius and case when len(alt) == 1, if statement is blee
    earth_rad = [6371000 for i in np.atleast_1d(alt)]
    vis_circle_rad = [satellite_basic(earth_rad[i], np.atleast_1d(alt)[i])[5] for i in
                      range(len(np.atleast_1d(earth_rad)))]
    if len(earth_rad) == 1:
        earth_rad = earth_rad[0]
        vis_circle_rad = vis_circle_rad[0]
    return lat, lon, alt, earth_rad, vis_circle_rad


def get_sat_positions_distance_away(time, path_to_tle, sat_name, distance):
    """Returns arrays of lat,lon of satellite that are away each other in given distance along sat propagation
    time, delta_time, num_of_div, tle_path
    Parameters
    ----------
    time : list
        List of times in skyfield.timescale format.
    path_to_tle : string
        Path to TLE file
    sat_name: string
        Name of satellite to be looked for.
    Returns
    -------
    lat: array
        Latitudes in array
    lon: array
        Longitude in array
    alt: array
        Height of satellite in meters.
    rad:
        Local earth radius in meters. TODO: right now constant radius
        :param distance:
    """

    from skyfield.api import load
    ts = load.timescale()

    start_time, end_time = time[0], time[-1]
    st = start_time.tai_calendar()
    # Points are generated one second away:
    dt = int((end_time - start_time) * 24 * 60 * 60)
    st_sec = int(st[5])
    lats, lons, alts, earth_rads, vis_circle_rads = ([] for i in range(5))
    positions = []
    positions.append(get_sat_positions(start_time, path_to_tle, sat_name))
    lat1, lon1 = positions[0][0], positions[0][1]

    times = ts.utc(st[0], st[1], st[2], st[3], st[4], range(st_sec, st_sec + dt))

    for i_time in times:
        position = get_sat_positions(i_time, path_to_tle, sat_name)
        lat2, lon2 = position[0], position[1]
        current_distance = find_distance_along_great_circle(lat1, lon1, lat2, lon2)
        if distance < current_distance:
            positions.append(position)
            lat1, lon1 = lat2, lon2


    for i in range(len(positions)):
        la, lo, al, er, vs = positions[i]
        lats.append(la)
        lons.append(lo)
        alts.append(al)
        earth_rads.append(er)
        vis_circle_rads.append(vs)

    print(lats)
    return lats, lons, alts, earth_rads, vis_circle_rads


def photo_of_area(hrit_files, central_lat, central_lon, elevation, time, dpi, photo_path, load_photo='VIS006',
                  all_lines=True):
    """Return path of photo
    Parameters
    ----------
    parameter_name : parameter_type
        Quick description of it.
    Returns
    -------
    photos: array
        Array of saved on disc files.

    """
    # Loading decompressed previously photos
    from satpy import Scene
    # from glob import glob
    import matplotlib.pyplot as plt
    from satpy import find_files_and_readers
    from datetime import datetime
    import cartopy.crs as ccrs
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = dpi
    # load_photo = 'VIS006'

    first = 0
    last = len(time) - 1
    # print(len(time),last)

    yearF, monthF, dayF, hourF, minuteF, secondF = time[first].tt_calendar()
    yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
    # IT IS NOT WORKING FIND SOLUTION - Adding a minute in case there is only one point on map
    if (len(time) == 1):
        time[last].tt = time[last].tt + 1 / 3600
        yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
        # print("udalosie")

    # print(yearF, monthF, dayF, hourF, minuteF, secondF )
    # print(yearL, monthL, dayL, hourL, minuteL, secondL)
    time[0].tt_calendar()[0]
    files = find_files_and_readers(base_dir=hrit_files,
                                   start_time=datetime(yearF, monthF, dayF, hourF, minuteF),
                                   end_time=datetime(yearL, monthL, dayL, hourL, minuteL),
                                   reader='seviri_l1b_hrit')
    scn = Scene(filenames=files)

    scn.load([load_photo])
    new_scn = scn
    crs = new_scn[load_photo].attrs['area'].to_cartopy_crs()

    div = 30
    y, x = polygon_surrounding_sat(central_lat, central_lon, elevation, div)
    extent = calculate_extent(x, y)
    ax = plt.axes(projection=ccrs.Sinusoidal(central_lon, central_lat))
    ax.set_extent(extent)

    if (all_lines):
        ax.gridlines()
        ax.coastlines(resolution='50m', color='red')
        ax.coastlines()

    # ax.gridlines()
    # ax.set_global()
    plt.imshow(new_scn[load_photo], transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    # cbar = plt.colorbar()
    # cbar.set_label("Kelvin")
    # plt.imsave('imsave.png', im, cmap='gray')
    name_img = photo_path + load_photo + "_" + datetime(yearF, monthF, dayF, hourF, minuteF).__str__() + '.png'
    plt.savefig(name_img, dpi=dpi)
    # print(name_img)

    return name_img


def calculate_extent(x, y):
    """Calculates extent of area by finding extreme coordinates of given set of points.
    Parameters
    ----------
    x : array
        Array of latitude coordinates.
    y : array
        Array of longitude coordinates.
    Returns
    -------
    extent: array
        For coordinates: [lat_min,lat_max,lon_min,lon_max]


    """
    import numpy as np

    div = len(x)
    lat_sort = np.sort(x, axis=0)
    lon_sort = np.sort(y, axis=0)
    extent = [lat_sort[0], lat_sort[div - 1], lon_sort[0], lon_sort[div - 1]]

    return extent


def set_of_photos(hrit_files, central_lat, central_lon, elevation, time, dpi, photo_path, all_lines=False,
                  load_photo=['VIS006']):
    """Return path of photo
    Parameters
    ----------
    parameter_name : parameter_type
        Quick description of it.
    Returns
    -------
    photos: array
        Array of saved on disc files.

    """
    # Loading decompressed previously photos
    from satpy import Scene
    import numpy as np
    # from glob import glob
    import matplotlib.pyplot as plt
    from satpy import find_files_and_readers
    from datetime import datetime
    import cartopy.crs as ccrs
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = dpi
    # load_photo = 'VIS006'

    first = 0
    last = len(time) - 1
    # print(len(time),last)

    yearF, monthF, dayF, hourF, minuteF, secondF = time[first].tt_calendar()
    yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
    # IT IS NOT WORKING FIND SOLUTION - Adding a minute in case there is only one point on map
    if len(time) == 1:
        time[last].tt = time[last].tt + 1 / 3600
        yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
        # print("udalosie")

    # print(yearF, monthF, dayF, hourF, minuteF, secondF )
    # print(yearL, monthL, dayL, hourL, minuteL, secondL)
    # time[0].tt_calendar()[0]
    files = find_files_and_readers(base_dir=hrit_files,
                                   start_time=datetime(yearF, monthF, dayF, hourF, minuteF),
                                   end_time=datetime(yearL, monthL, dayL, hourL, minuteL),
                                   reader='seviri_l1b_hrit')
    scn = Scene(filenames=files)
    load_photo = scn.available_dataset_names()
    # HRV does not work properly
    load_photo.remove('HRV')
    # print(load_photo)
    # print(scn.available_dataset_names())

    div = 30
    name_img = []
    for photo_type in load_photo:
        # print(photo_type)
        scn.load([photo_type])
        # print(scn)
        for i in range(len(time)):
            new_scn = scn
            # print(new_scn)
            crs = new_scn[photo_type].attrs['area'].to_cartopy_crs()
            y, x = polygon_surrounding_sat(central_lat[i], central_lon[i], elevation[i], div)
            extent = calculate_extent(x, y)
            ax = plt.axes(projection=ccrs.Sinusoidal(central_lon[i], central_lat[i]))
            ax.set_extent(extent)

            if (all_lines):
                ax.gridlines()
                ax.coastlines(resolution='50m', color='red')
                ax.coastlines()

            # ax.gridlines()
            # ax.set_global()
            plt.imshow(new_scn[photo_type], transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
            # cbar = plt.colorbar()
            # cbar.set_label("Kelvin")
            # plt.imsave('imsave.png', im, cmap='gray')
            name = photo_path + photo_type + "_" + time[i].utc_iso() + '.png'
            plt.savefig(name, dpi=dpi)
            name_img = np.append(name_img, name)

    return (name_img)


def show_overall(hrit_files, central_lat, central_lon, elevation, time, dpi, photo_path, all_lines=False,
                 pro='PlateCarree', load_photo=['VIS006'], save=False):
    """Return path of photo
    Parameters
    ----------
    pro : string
        PlateCarree
        Sinusoidal
    Returns
    -------
    photos: array
        Array of saved on disc files.

    """
    from satpy.scene import Scene
    from satpy import find_files_and_readers
    from datetime import datetime
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    # %matplotlib inline
    import matplotlib.pyplot as plt

    ### START - CALCULATES OVERALL EXTENT ###
    div = 30
    x, y = [], []
    for i in range(len(central_lat)):
        y_temp, x_temp = polygon_surrounding_sat(central_lat[i], central_lon[i], elevation[i], div)
        x = np.append(x, x_temp)
        y = np.append(y, y_temp)

    extent = calculate_extent(x, y)
    ### END - CALCULATES OVERALL EXTENT ###

    first = 0
    last = len(time) - 1

    yearF, monthF, dayF, hourF, minuteF, secondF = time[first].tt_calendar()
    yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
    if len(time) == 1:
        time[last].tt = time[last].tt + 1 / 3600
        yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()

    # print(yearF, monthF, dayF, hourF, minuteF, secondF )
    # print(yearL, monthL, dayL, hourL, minuteL, secondL)
    # time[0].tt_calendar()[0]
    files = find_files_and_readers(base_dir=hrit_files,
                                   start_time=datetime(yearF, monthF, dayF, hourF, minuteF),
                                   end_time=datetime(yearL, monthL, dayL, hourL, minuteL),
                                   reader='seviri_l1b_hrit')
    scn = Scene(filenames=files)
    scn.load(load_photo)
    new_scn = scn
    crs = new_scn[load_photo[0]].attrs['area'].to_cartopy_crs()
    # ax = plt.axes(projection=crs)
    central_lon, central_lat = (extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2

    if pro == 'PlateCarree':
        ax = plt.axes(projection=ccrs.PlateCarree())
        pro_name = '_PlateCarree_'
        ax.set_global()
    if pro == 'Sinusoidal':
        ax = plt.axes(projection=ccrs.Sinusoidal(central_lon, central_lat))
        ax.set_extent(extent)
        pro_name = '_Sinusoidal_'

    if all_lines:
        ax.coastlines(color='red')
        ax.gridlines(color='red')
        ax.add_feature(cfeature.BORDERS, color='red')
        pro_name = pro_name + 'lines_'

    plt.imshow(new_scn[load_photo[0]], transform=crs, extent=crs.bounds, origin='upper', cmap='gray')

    if save:
        if isinstance(load_photo[0], float):
            photo_type = str(load_photo[0])
        else:
            photo_type = load_photo[0]
        name = photo_path + photo_type + pro_name + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(date=scn.start_time)
        plt.savefig(name, dpi=dpi)
    if not save:
        plt.show()

    return ()


def simple_export(hrit_files, time, dpi, photo_path, load_photo=['VIS006']):
    from satpy import Scene
    from satpy import find_files_and_readers
    from datetime import datetime
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = dpi
    # load_photo = 'VIS006'

    first = 0
    last = len(time) - 1
    # print(len(time),last)

    yearF, monthF, dayF, hourF, minuteF, secondF = time[first].tt_calendar()
    yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
    # IT IS NOT WORKING FIND SOLUTION - Adding a minute in case there is only one point on map
    if len(time) == 1:
        time[last].tt = time[last].tt + 1 / 3600
        yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
        # print("It works")

    # print(yearF, monthF, dayF, hourF, minuteF, secondF )
    # print(yearL, monthL, dayL, hourL, minuteL, secondL)
    # time[0].tt_calendar()[0]
    files = find_files_and_readers(base_dir=hrit_files,
                                   start_time=datetime(yearF, monthF, dayF, hourF, minuteF),
                                   end_time=datetime(yearL, monthL, dayL, hourL, minuteL),
                                   reader='seviri_l1b_hrit')
    scn = Scene(filenames=files)

    scn.load(load_photo)
    file = photo_path + 'globe_' + load_photo[0] + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(date=scn.start_time)
    scn.save_dataset(load_photo[0], writer='simple_image', filename=file, num_threads=8)

    return ()


def show_sat_perspective(hrit_files, central_lat, central_lon, elevation, time, dpi, save_path, fov, composite=None):
    """Shows in Jupyter Notebook results of pictures seen from sat
    Parameters
        Array of saved on disc files.
        :param save_path:
        :param composite:
        :param dpi:
        :param time:
        :param elevation:
        :param central_lon:
        :param central_lat:
        :param hrit_files:

    """
    # TO DO: Add local earth radius
    if composite is None:
        composite = 'realistic_colors'
    import datetime as dt
    from satpy.scene import Scene
    from satpy.resample import get_area_def
    from datetime import datetime
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import cartopy.feature as cfeature
    from skyfield.api import Topos, load
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import Angle
    # %matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from pyresample.geometry import AreaDefinition, create_area_def

    area_def = []
    for i in range(0, len(central_lon)):
        area_id = 'ease_sh'
        center = (central_lat[i], central_lon[i])
        radius = satellite_info(6371228, elevation[i], fov[0], fov[1])[5]
        resolution = 2500
        proj_string = '+proj=laea +lat_0=' + np.array2string(central_lat[i]) + ' +lon_0=' + np.array2string(
            central_lon[i]) + ' +a=6371228.0 +units=m'
        area_def.append(create_area_def(area_id, proj_string, center=center, radius=radius, resolution=resolution))

    files = return_files(time, hrit_files)

    scn = Scene(filenames=files)
    scn.load([composite])
    new_scn = scn

    for i, area_def in enumerate(area_def, start=0):
        local_scn = scn.resample(area_def, radius_of_influence=50000)
        local_scn.show(composite)
        path = save_path + composite + '_' + str(i) + '.png'
        local_scn.save_dataset(composite, path, writer='simple_image', num_threads=8)

    if save_path:
        if (isinstance(load_photo[0], float)):
            photo_type = str(load_photo[0])
        else:
            photo_type = load_photo[0]
        name = photo_path + photo_type + pro_name + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(date=scn.start_time)
        plt.savefig(name, dpi=dpi)
    if not save_path:
        plt.show()

    return ()


def return_files(time, hrit_files):
    from satpy import find_files_and_readers
    from datetime import datetime

    first = 0
    last = len(time) - 1
    yearF, monthF, dayF, hourF, minuteF, secondF = time[first].tt_calendar()
    yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()
    if len(time) == 1:
        time[last].tt = time[last].tt + 1 / 3600
        yearL, monthL, dayL, hourL, minuteL, secondL = time[last].tt_calendar()

    files = find_files_and_readers(base_dir=hrit_files,
                                   start_time=datetime(yearF, monthF, dayF, hourF, minuteF),
                                   end_time=datetime(yearL, monthL, dayL, hourL, minuteL),
                                   reader='seviri_l1b_hrit')

    return files


def show_sat_perspective2(hrit_files, central_lat, central_lon, elevation, time, save_path, fov, shape, composite=None,
                          fov_deg=True):
    """Shows in Jupyter Notebook results of pictures seen from sat
    Parameters
        Array of saved on disc files.
        :param save_path:
        :param composite:
        :param time:
        :param elevation:
        :param central_lon:
        :param central_lat:
        :param hrit_files:

    """
    # TO DO: Add local earth radius
    import datetime as dt
    from satpy.scene import Scene
    import math
    from satpy.resample import get_area_def
    from datetime import datetime
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import cartopy.feature as cfeature
    from skyfield.api import Topos, load
    import pyproj
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import Angle
    # %matplotlib inline
    import matplotlib.pyplot as plt
    from pyresample.geometry import AreaDefinition, SwathDefinition, create_area_def

    if composite is None:
        composite = 'realistic_colors'

    if fov_deg == True:
        fov = [fov[0] * math.pi / 180, fov[1] * math.pi / 180]

    area_def = []
    for i in range(0, len(central_lon)):
        area_id = 'ease_sh'
        rad = satellite_info(6371228, elevation[i], fov[0], fov[1])[5] / 2
        lat_0, lon_0 = central_lat[i], central_lon[i]
        lat_1, lat_2 = central_lat[i], central_lon[i]
        center = (central_lat[i], central_lon[i])
        radius = (rad, rad)
        area_id = 'wrf_circle'
        proj_dict = {'proj': 'lcc', 'lat_0': lat_0, 'lon_0': lon_0, \
                     'lat_1': lat_1, 'lat_2': lat_2, \
                     'a': 6370000, 'b': 6370000}
        area_def.append(AreaDefinition.from_circle(area_id, proj_dict, center, radius, shape=shape))

    files = return_files(time, hrit_files)

    scn = Scene(filenames=files)
    scn.load([composite])
    new_scn = scn

    for i, area_def in enumerate(area_def, start=0):
        local_scn = scn.resample(area_def, radius_of_influence=50000)
        local_scn.show(composite)
        path = save_path + str(shape[0]) + '_' + str(composite) + '_' + str(i) + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(
            date=scn.start_time)
        local_scn.save_dataset(composite, path, writer='simple_image', num_threads=8)

    # if save_path:
    #     if (isinstance(load_photo[0], float)):
    #         photo_type = str(load_photo[0])
    #     else:
    #         photo_type = load_photo[0]
    #     name = photo_path + photo_type + pro_name + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(date=scn.start_time)
    #     plt.savefig(name, dpi=dpi)
    # if not save_path:
    #     plt.show()

    return ()


def show_sat_perspective3(hrit_files, central_lat, central_lon, elevation, time, save_path, fov, shape, proj,
                          projection_parameters, composite=None, fov_deg=True):
    """Shows in Jupyter Notebook results of pictures seen from sat
    Parameters
        Array of saved on disc files.
        :param save_path:
        :param composite:
        :param time:
        :param elevation:
        :param central_lon:
        :param central_lat:
        :param hrit_files:

    """
    # TO DO: Add local earth radius
    from satpy.scene import Scene
    import math
    from pyresample.geometry import AreaDefinition, SwathDefinition, create_area_def
    from pyresample import create_area_def

    if composite is None:
        composite = 'realistic_colors'

    if fov_deg == True:
        fov = [fov[0] * math.pi / 180, fov[1] * math.pi / 180]

    # lla = mat.find_sourounding_list(earth_rads, latitudes, longitudes, elevations, fov)

    area_def = []
    for i in range(0, len(central_lon)):
        altitude = elevation[i]
        rad = satellite_info(6371228, elevation[i], fov[0], fov[1])[5] / 2
        lat_0, lon_0 = central_lat[i], central_lon[i]
        lat_1, lat_2 = central_lat[i], central_lon[i]
        center = (central_lat[i], central_lon[i])
        radius = (rad, rad)
        area_id = 'wrf_circle'
        proj_dict = {'proj': proj, 'lat_0': lat_0, 'lon_0': lon_0, \
                     'lat_1': lat_1, 'lat_2': lat_2, \
                     'a': 6370000, 'b': 6370000, 'h': altitude, 'azi': projection_parameters[0],
                     'tilt': projection_parameters[1]}
        area_def.append(AreaDefinition.from_circle(area_id, proj_dict, center, radius=radius, shape=shape))
        # area_def.append(AreaDefinition.create_area_def(area_id, proj_dict, center, radius=radius, shape=shape))

    files = return_files(time, hrit_files)

    scn = Scene(filenames=files)
    scn.load([composite])
    new_scn = scn

    for i, area_def in enumerate(area_def, start=0):
        local_scn = scn.resample(area_def, radius_of_influence=50000)
        local_scn.show(composite)
        path = save_path + proj + str(projection_parameters[0]) + '_' + str(projection_parameters[1]) + '_' + str(
            shape[0]) + '_' + str(composite) + '_' + '_{date:%Y-%m-%d_%H_%M_%S}'.format(
            date=scn.start_time) + '/' + str(i) + '.png'
        local_scn.save_dataset(composite, path, writer='simple_image', num_threads=8)

    # if save_path:
    #     if (isinstance(load_photo[0], float)):
    #         photo_type = str(load_photo[0])
    #     else:
    #         photo_type = load_photo[0]
    #     name = photo_path + photo_type + pro_name + '_{date:%Y-%m-%d_%H_%M_%S}.png'.format(date=scn.start_time)
    #     plt.savefig(name, dpi=dpi)
    # if not save_path:
    #     plt.show()

    return ()


def calculate_and_project(hrit_files, sat_positions, time, save_path, fov, shape, proj,
                          nadir_proj=True, composite=None, fov_deg=True, save_data_path=None, save_photos = True):
    """Shows in Jupyter Notebook results of pictures seen from sat
    Parameters
        Array of saved on disc files.
        :param save_path:
        :param composite:
        :param time:
        :param elevation:
        :param central_lon:
        :param central_lat:
        :param hrit_files:

    """
    # TO DO: Add local earth radius
    from satpy.scene import Scene
    from wutsat.fun import mat_fun
    import math
    from pyresample.geometry import AreaDefinition, SwathDefinition, create_area_def
    # from pyresample import create_area_def
    import os

    if composite is None:
        composite = 'realistic_colors'

    if fov_deg == True:
        fov = [fov[0] * math.pi / 180, fov[1] * math.pi / 180]

    if nadir_proj:
        nadir_proj = [0,0]

    central_lat, central_lon, elevation = mat_fun.find_sourounding_list(earth_radius=sat_positions[3],
                                                                        lat=sat_positions[0], lon=sat_positions[1],
                                                                        alt=sat_positions[2], fov=fov)

    #print(len(central_lat))
    area_def = []
    for i in range(0, len(central_lon)):
        altitude = elevation[i]
        rad = satellite_info(6371228, elevation[i], fov[0], fov[1])[5] / 2
        lat_0, lon_0 = central_lat[i], central_lon[i]
        lat_1, lat_2 = central_lat[i], central_lon[i]
        center = (central_lat[i], central_lon[i])
        radius = (rad, rad)
        area_id = 'wrf_circle'
        proj_dict = {'proj': proj, 'lat_0': lat_0, 'lon_0': lon_0, \
                     'lat_1': lat_1, 'lat_2': lat_2, \
                     'a': 6370000, 'b': 6370000, 'h': altitude, 'azi': nadir_proj[0],
                     'tilt': nadir_proj[1]}
        area_def.append(AreaDefinition.from_circle(area_id, proj_dict, center, radius=radius, shape=shape))
        # area_def.append(AreaDefinition.create_area_def(area_id, proj_dict, center, radius=radius, shape=shape))

    files = return_files(time, hrit_files)

    if save_photos:
        scn = Scene(filenames=files)
        scn.load([composite])
        for i, area in enumerate(area_def, start=0):
            local_scn = scn.resample(area, radius_of_influence=50000)
            local_scn.show(composite)
            path = save_path + '/' + str(i) + '.png'
            local_scn.save_dataset(composite, path, writer='simple_image', num_threads=8)
    sat_data = [area_def, files, [central_lat, central_lon, elevation]]
    if save_data_path:
        mat_fun.rwdata(save_data_path, 'sat_data.pkl', 'w', sat_data)

    return ()


def get_area(hrit_files, central_lat, central_lon, elevation, time, save_path, fov, shape, proj,
             projection_parameters, composite=None, fov_deg=True):
    """Shows in Jupyter Notebook results of pictures seen from sat
    Parameters
        Array of saved on disc files.
        :param save_path:
        :param composite:
        :param time:
        :param elevation:
        :param central_lon:
        :param central_lat:
        :param hrit_files:

    """
    # TO DO: Add local earth radius
    from satpy.scene import Scene
    import math
    from pyresample.geometry import AreaDefinition, SwathDefinition, create_area_def
    from pyresample import create_area_def

    if composite is None:
        composite = 'realistic_colors'

    if fov_deg == True:
        fov = [fov[0] * math.pi / 180, fov[1] * math.pi / 180]

    area_def = []
    for i in range(0, len(central_lon)):
        altitude = elevation[i]
        rad = satellite_info(6371228, elevation[i], fov[0], fov[1])[5] / 2
        lat_0, lon_0 = central_lat[i], central_lon[i]
        lat_1, lat_2 = central_lat[i], central_lon[i]
        center = (central_lat[i], central_lon[i])
        radius = (rad, rad)
        area_id = 'wrf_circle'
        proj_dict = {'proj': proj, 'lat_0': lat_0, 'lon_0': lon_0, \
                     'lat_1': lat_1, 'lat_2': lat_2, \
                     'a': 6370000, 'b': 6370000, 'h': altitude, 'azi': projection_parameters[0],
                     'tilt': projection_parameters[1]}
        area_def.append(AreaDefinition.from_circle(area_id, proj_dict, center, radius=radius, shape=shape))
        # area_def.append(AreaDefinition.create_area_def(area_id, proj_dict, center, radius=radius, shape=shape))

    files = return_files(time, hrit_files)

    return (area_def, files, composite)


def show_proj(latitudes, longitudes, angle, extent, proj, label=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.collections as mplcollections
    from matplotlib.patches import Circle
    import cartopy.crs as ccrs
    from math import degrees as to_deg
    # Ustawianie kolorów
    colors = np.linspace(0, 1, num=5)
    if extent == None:
        extent = [0, 0, 0, 0]
    # Ustawianie rzutowania
    if proj == 'orto':
        bmap = ccrs.Orthographic(central_longitude=((extent[0] + extent[1]) / 2),
                                 central_latitude=((extent[2] + extent[3]) / 2))
    if proj == 'albers':
        bmap = ccrs.AlbersEqualArea(central_longitude=0)
    if proj == 'sinus':
        bmap = ccrs.Sinusoidal(central_longitude=((extent[0] + extent[1]) / 2))
    if proj == 'merc':
        bmap = ccrs.Mercator(central_longitude=((extent[0] + extent[1]) / 2))
    if proj == 'plate':
        bmap = ccrs.PlateCarree(central_longitude=((extent[0] + extent[1]) / 2))

    ax = plt.axes(projection=bmap)
    ax.set_global()
    ax.coastlines()
    ax.gridlines()
    if not extent == [0, 0, 0, 0]:
        ax.set_extent(extent)

    # Ładowanie kół opisujących maksymalną widoczność
    patches = []
    for i in range(len(latitudes)):
        patches.append(Circle((latitudes[i], longitudes[i]), radius=to_deg(angle[i])))

    collection = mplcollections.PatchCollection(patches,
                                                transform=ccrs.PlateCarree())
    # Rysowanie
    collection.set_array(colors)
    ax.add_collection(collection)
    ax.set_title(label)
    plt.savefig('CircleTest.png', dpi=100)
    plt.show()
    return ()

def get_extent_in_coordinates(extent, area_def, files, composite='realistic_colors'):
    """Returns list of lats,lons coresponding to the pixel points of given extent
    Parameters:
    """

    # TODO: Cheeck this fukin shit;
    from satpy.scene import Scene
    # import math
    # from pyresample.geometry import AreaDefinition, SwathDefinition, create_area_def

    # files = return_files(time, hrit_files)

    scn = Scene(filenames=files)
    scn.load([composite])
    new_scn = scn
    local_scn = scn.resample(area_def, radius_of_influence=50000)
    lons, lats = local_scn[composite].attrs['area'].get_lonlats()
    ext_lats, ext_lons = ([], [])

    for i in range(len(extent)):
        x = extent[i][0][1].astype(int)
        y = extent[i][0][0].astype(int)
        if 0 <= x <= len(lats) and 0 <= y <= len(lats):
            ext_lats.append(lats[x][y])
            ext_lons.append(lons[x][y])

    return ext_lats, ext_lons

def get_photo_extent_and_points(area_def, files, coposite):
    ''' Returns
    :param area_def:
    :param files:
    :param coposite:
    :return:
    '''
    import numpy as np

    max_size = area_def.shape[0] - 1
    bound = np.array([[[0,0]],[[0,max_size]],[[max_size,max_size]],[[max_size,0]]])
    photo_points = get_extent_in_coordinates(bound, area_def, files, coposite)
    ph_ext_lats, ph_ext_lons = photo_points[0], photo_points[1]
    photo_extent = [min(ph_ext_lons), max(ph_ext_lons), min(ph_ext_lats), max(ph_ext_lats)]

    return photo_extent, photo_points

def get_timeframe_of_sat_positions(points, path_to_tle, sat_name, time, distance,plot=False):
    """Returns satellite start and end position depending on given point that it could see along path taken from TLE
    Parameters:
    """
    # TODO: Check or calculate time such sat makes max half orbit, otherwise it it might not make any more sens
    from skyfield.api import load
    ts = load.timescale()

    start_time, end_time = time[0], time[-1]
    st = start_time.utc
    # Points are generated one second away:
    dt = int((end_time - start_time) * 24 * 60 * 60)
    st_sec = int(st[5])
    times = ts.utc(st[0], st[1], st[2], st[3], st[4], range(st_sec, st_sec + dt))

    positions = get_sat_positions(times, path_to_tle, sat_name)
    dst1,dst2,pnt1,pnt2=[],[],[],[]
    for i, (lat2, lon2) in enumerate(zip(positions[0],positions[1])):
        current_distance = [find_distance_along_great_circle(point[0], point[1], lat2, lon2) for point in points]
        dst1.append(current_distance.copy())
        pnt1.append(points.copy())
        if all(x <= distance for x in current_distance):
            pos0 = [lat2,lon2]
            time0 = times[i]
            st = i
            print('Gorne koordynaty: \n', pos0,current_distance)
            break
        del current_distance


    for i, (lat2, lon2) in reversed(list(enumerate(zip(positions[0],positions[1])))):
        current_distance = [find_distance_along_great_circle(point[0], point[1], lat2, lon2) for point in points]
        dst2.append(current_distance.copy())
        pnt2.append(points.copy())
        if all(x <= distance for x in current_distance):
            pos1 = [lat2, lon2]
            time1 = times[i]
            end = i
            print('Dolne koordynaty: \n', pos1)
            break
    # all = [dst1, dst2, time0, time1, st, end,positions]

    # lat_arr = [np.append(positions[0][0:st],positions[0][end:-1]), positions[0][st:end], [points[0]]]
    # lon_arr = [np.append(positions[1][0:st],positions[1][end:-1]), positions[1][st:end], [points[1]]]
    # if plot:
    #     pr.plot_multi_list_of_points_on_map(lat_arr,lon_arr,'Zakres widoczności przez satelite danego punktu')
    #     #plotting_presets.plot_list_of_points_on_map(positions[0][st:end],positions[1][st:end],'Wszystkie punkty')
    times = [time0, time1]
    points = [[pos0[0],pos1[0]], [pos0[1],pos1[1]]]
    polygons = calculate_sets_of_footprints(points[0],points[1],[distance,distance],[6371000,6371000])
    debug = [positions, times, [st,end], [pos0,pos1],[dst1,dst2]]
    return times, points, polygons, debug

def calculate_footprint(lat,lon,distance, radius=6371000,resolution=200):
    import math as m

    lats, lons = [], []
    steps = [2 * m.pi * i /resolution for i in range(resolution + 1)]
    for step in steps:
        la,lo  = find_point(lat, lon, step, distance, radius=radius)
        lats.append(la)
        lons.append(lo)

    return lats,lons

def calculate_sets_of_footprints(lats,lons,distances,radiuses,resolution=200):
    #TODO: Add this to tests:
    if not len(lats) == len(lons) == len(distances) == len(radiuses):
        raise NameError('Wrong input data')

    set_lats, set_lons = [], []
    for la, lo, dist, rad in zip(lats,lons,distances,radiuses):
        lat, lon = calculate_footprint(la,lo,dist,rad,resolution)
        set_lats.append(lat)
        set_lons.append(lon)

    return set_lats, set_lons
