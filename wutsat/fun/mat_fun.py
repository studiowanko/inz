def find_angle_analytic(r, h, fi):
    """Finds an angle from nadir to point that is intersection line that goes through circle.
    Parameters
    ----------
    r : float
        Radious of circle
    h : float
        height from circle to point where satellite is.
    fi : float
        FOV / 2 of sensor that looks at circle in radians
    Returns
    -------
    beta: float
        Angle in radians of arc.
    """
    import math as m
    H = h + r
    A = (-1 / m.tan(fi))

    a = m.pow(A, 2) + 1
    b = 2 * A * H
    c = m.pow(H, 2) - m.pow(r, 2)
    d = m.pow(b, 2) - (4 * a * c)
    if d < 0:
        print("Error delta less than zero: %", d)
        raise ValueError('Math error!')
    x1 = (-b - m.sqrt(d)) / (2 * a)
    x2 = (-b + m.sqrt(d)) / (2 * a)
    y11 = A * x1 + H
    y12 = m.sqrt(m.pow(r, 2) - m.pow(x1, 2))
    y21 = A * x2 + H
    y22 = m.sqrt(m.pow(r, 2) - m.pow(x2, 2))

    if y11 < 0 or y12 < 0 or y21 < 0 or y22 < 0 or x1 < 0 or x2 < 0:
        # TODO: Fix this assumptions -> they are not correct here
        print('Checks math assumptions error! x1 = %f x2 = %f', x1, x2)
        # raise ValueError('Math error!')
    x = min(x1, x2)
    y = m.sqrt(m.pow(r, 2) - m.pow(x, 2))
    beta = m.pi / 2 - m.atan(y / x)
    return beta


def find_sourounding(sat_info, earth_radius, lat, lon, alt):
    """It creates a list of lat,lon,alt that are multiply of radius of what camera with a given dFOV would see
    away from from a given lat,lon
    Parameters
    ----------
    sat_info : list
        List of satellite information. See sat_info function for more details.
    earth_radius : list
        List of satellite information. See sat_info function for more details.
    lat : float
        Latitude of location to be calculated.
    lon : float
        Longitude of location to be calculated.
    alt : float
        Altitude of location to be calculated.
    Returns
    -------
    new_lats: list of floats
        List of latitudes.
    new_lons: list of floats
        List of longitudes.
    new_alts: list of floats
        List of corresponding altitudes.
    """
    import math as m
    from wutsat.fun import sat_fun as s
    new_lats, new_lons, new_alts = [], [], []
    n = m.ceil(sat_info[4] / sat_info[8])
    vis_rad = sat_info[5] / 5
    dis = list(range(1, n))
    dis = [i * vis_rad for i in dis]

    for distance in dis:
        angles = list(range(0, m.ceil(m.pi * distance / vis_rad)))
        angles = [i * 2 * m.pi / m.ceil(m.pi * distance / vis_rad) for i in angles]
        for angle in angles:
            new_lat, new_lon = s.find_point(lat, lon, angle, distance, earth_radius)
            new_lats.append(new_lat)
            new_lons.append(new_lon)
            new_alts.append(alt)
    return new_lats, new_lons, new_alts


def find_sourounding_list(earth_radius, lat, lon, alt, fov):
    import operator
    import functools
    import numpy as np

    list_of_lats, list_of_lons, list_of_alts = [], [], []

    for ra, la, lo, al in zip(earth_radius, lat, lon, alt):
        # print(lats, lons, alts)
        # TODO: comment out 3 lines below?! multiple lats ect. are duplicated
        list_of_lats.append([float(la)])
        list_of_lons.append([float(lo)])
        list_of_alts.append([float(al)])
        new_lats, new_lons, new_alts = find_sourounding_points(ra, la, lo, al, fov)
        list_of_lats.append(new_lats)
        list_of_lons.append(new_lons)
        list_of_alts.append(new_alts)
    list_of_lats = np.asarray(functools.reduce(operator.concat, list_of_lats))
    list_of_lons = np.asarray(functools.reduce(operator.concat, list_of_lons))
    list_of_alts = np.asarray(functools.reduce(operator.concat, list_of_alts))
    return list_of_lats, list_of_lons, list_of_alts


def find_sourounding_points(earth_radius, lat, lon, alt, fov):
    """It creates a list of lat,lon,alt that are multiply of radius of what camera with a given dFOV would see
    away from from a given lat,lon
    Parameters
    ----------
    sat_info : list
        List of satellite information. See sat_info function for more details.
    earth_radius : list
        List of satellite information. See sat_info function for more details.
    lat : float
        Latitude of location to be calculated.
    lon : float
        Longitude of location to be calculated.
    alt : float
        Altitude of location to be calculated.
    Returns
    -------
    new_lats: list of floats
        List of latitudes.
    new_lons: list of floats
        List of longitudes.
    new_alts: list of floats
        List of corresponding altitudes.
    """
    import math as m
    from wutsat.fun import sat_fun as s
    sat_info = s.satellite_info(earth_radius, alt, fov[0], fov[1])
    new_lats, new_lons, new_alts = [], [], []
    n = m.ceil(sat_info[4] / sat_info[8])
    vis_rad = sat_info[5] / 5
    dis = list(range(1, n))
    dis = [i * vis_rad for i in dis]

    for distance in dis:
        angles = list(range(0, m.ceil(m.pi * distance / vis_rad)))
        angles = [i * 2 * m.pi / m.ceil(m.pi * distance / vis_rad) for i in angles]
        for angle in angles:
            new_lat, new_lon = s.find_point(lat, lon, angle, distance, earth_radius)
            new_lats.append(new_lat)
            new_lons.append(new_lon)
            new_alts.append(alt)
    return new_lats, new_lons, new_alts


def get_evenly_spaced_circles():
    x, y = ([],) * 2
    for i in range(5):
        for j in range(11):
            a = -150 + j * 30
            b = -60 + i * 30
            x = x + [a]
            y = y + [b]
    r = [0.2 for i in range(len(x))]
    return (x, y, r)


def intersection(line1, line2):
    """
    #TODO: WRITE IT XD
    """
    from shapely.geometry import LineString
    line = LineString(line1)
    other = LineString(line2)

    return line.intersects(other)


def check_all_methods(methods, area_min=20, area_max=9990):
    """
    Checks all methods and returns when convex quad exists and area is at least x percent
    """
    res1, res2 = ([], [])
    for i, met in enumerate(methods):
        for j, m in enumerate(met):
            extent = m[2]
            line1 = [(extent[0][0][0], extent[0][0][1]), (extent[2][0][0], extent[2][0][1])]
            line2 = [(extent[1][0][0], extent[1][0][1]), (extent[3][0][0], extent[3][0][1])]
            is_convex_quad = intersection(line1, line2)
            # print(is_convex_quad)
            if is_convex_quad:
                # print([i, j])
                if area_max > calculate_relative_area(m) > area_min:
                    ii, jj = i, j
                    res1.append(ii)
                    res2.append(jj)

    return [res1, res2]


def calculate_relative_area(points):
    # TODO: NAME IT
    x = []
    y = []
    [x.append(xs[0][0]) for xs in points[2]]
    [y.append(ys[0][1]) for ys in points[2]]
    MatchArea = PolyArea(x, y)
    ImageArea = points[3][1][0] * points[3][1][1]
    RelativeArea = MatchArea / ImageArea * 100
    return RelativeArea


def PolyArea(x, y):
    # TODO: NAME IT
    # Clalculates area of polygon with given x and y
    import numpy as np
    return (0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def rwdata(path, file, mode, data):
    import os
    import pickle
    file = os.path.join(path, file)
    # if not os.path.exists(file):
    if mode == 'w':
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    elif mode == 'r':
        with open(file, 'rb') as f:
            data = pickle.load(f)
    return data


def draw_polygons_calculate_extent(pols, lines):
    """Return: extent min_x, max_x, min_y, max_y"""
    extent = []
    for data in pols + lines:
        lon_min, lon_max = min(min(data[0])), max(max(data[0]))
        lat_min, lat_max = min(min(data[1])), max(max(data[1]))
        extent.append([lon_min, lon_max, lat_min, lat_max])
    return extent


def calculate_results(t_in, t_out, v):
    """Return: extent min_x, max_x, min_y, max_y"""
    t_in_diff = t_in[-1] - t_in[0]
    t_out_diff = t_out[-1] - t_out[0]
    t_in_min = t_in_diff * 24 * 60 * 60
    t_out_min = t_out_diff * 24 * 60 * 60

    diff_percentage = (t_in_diff - t_out_diff) / t_in_diff * 100
    results = "Dla kompozytu {} i setu zdjec {}\n".format(v.composite, v.set_id)
    results = results + "Czas wejściowy: %d\n" % t_in_min
    results = results + "Czas wyjściowy: %d\n" % t_out_min
    results = results + "Różnica porcentowa: %d%%\n" % diff_percentage

    return results


def save_results(data, path, other_data):
    import os
    file = 'results.txt'
    file = os.path.join(path, file)
    data = data + "Ilosc wykrytych pasujacych zdjec : {}\n".format(str(len(other_data[0])))
    data = data + "Info drugie : {}".format(other_data[1])

    # if not os.path.exists(file):
    text_file = open(file, "wt")
    n = text_file.write(data)
    text_file.close()

    return
