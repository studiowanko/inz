import math
from wutsat.fun.parse import initialize_photo, initialize_projection, initialize_extra_info
import os


# Init photos_ds and projection_ds
core = "/mnt/c/inz/"
photos_ds_path = core + 'inz/data/photos.csv'
projection_ds_path = core + 'inz/data/data_sets.csv'

photos_ds = initialize_photo(photos_ds_path, core)
projection_ds = initialize_projection(projection_ds_path, core)
info_ds = initialize_extra_info(projection_ds_path,photos_ds_path, core)
pwsat2_tle_path = core + 'inz/data/pw-sat2_tle.txt'  ### Path to TLE file
hrit_files = core + 'hrit/decompressed-all/'  ### PATH to hrit decompressed files

hfov, vfov = 80 * math.pi / 180, 70 * math.pi / 180
pwsat_fov = [hfov, vfov]
pwsat2 = '/mnt/c/inz/pwsat2/'

def end_sound():
    os.system("sleep 0.5")
    os.system("echo -ne '\007'")
    os.system("sleep 0.5")
    os.system("echo -ne '\007'")
    os.system("sleep 0.5")
    os.system("echo -ne '\007'")
