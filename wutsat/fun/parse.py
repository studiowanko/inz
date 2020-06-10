def parse_start_time(x):
    from skyfield.api import load
    import dateutil.parser
    ts = load.timescale()
    return ts.utc(dateutil.parser.parse(x.start_time)).tt


def parse_end_time(x):
    from skyfield.api import load
    import dateutil.parser
    ts = load.timescale()
    return ts.utc(dateutil.parser.parse(x.end_time)).tt


def initialize_photo(dataset, core_path):
    import pandas as pd

    photos_data = pd.read_csv(dataset)
    photos_data.head()

    photos_data['star_t_tt'] = photos_data.apply(parse_start_time, axis=1)
    photos_data['end_t_tt'] = photos_data.apply(parse_end_time, axis=1)
    photos_data["source_img"] = core_path + "img/" + photos_data["img"]
    return photos_data


def initialize_projection(dataset, core_path):
    import pandas as pd
    ds = pd.read_csv(dataset)
    ds['general_path'] = core_path + "results/" + ds['set_id'].astype(str) + "_" + ds['name']
    ds['projected_set_path'] = ds["general_path"] + "/" + ds["composite"] + ds["projection"] + ds["dpi"].astype(str)
    ds["results_path"] = ds["general_path"] + "/" + "results/"
    ds["shape"] = [(3000, 3000)] * len(ds)
    ds["composite_plot"] = 0.6
    ds["res_im1"] = ds['general_path'] + "/im1.jpg"
    ds["res_im3"] = ds['general_path'] + "/im3.jpg"
    ds["res_im2"] = ds['general_path'] + "/im2.jpg"
    ds["res_im4"] = ds['general_path'] + "/im4.jpg"
    ds["res_im5"] = ds['general_path'] + "/im5.jpg"
    ds["res_final"] = ds['general_path'] + "/res_final.png"
    return ds


def initialize_extra_info(dataset_1, dataset_2, core_path):
    import pandas as pd
    ds_1 = pd.read_csv(dataset_2)
    ds_2 = pd.read_csv(dataset_1)

    ds_1['star_t_tt'] = ds_1.apply(parse_start_time, axis=1)
    ds_1['end_t_tt'] = ds_1.apply(parse_end_time, axis=1)
    ds_1["source_img"] = core_path + "img/" + ds_1["img"]

    return ds_1


class InitVariables:
    def __init__(self, i):
        from wutsat.fun import config as cf
        from skyfield.api import load
        import numpy as np
        import os

        ts = load.timescale()

        self.photos_ds = cf.photos_ds
        self.projection_ds = cf.projection_ds
        self.info_ds = cf.info_ds
        self.set_id = self.projection_ds["set_id"][i]
        self.start_time = self.photos_ds["star_t_tt"][self.set_id]
        self.end_time = self.photos_ds["end_t_tt"][self.set_id]
        self.TLE = self.projection_ds['tle_name'][i]
        self.dpi = self.projection_ds['dpi'][i]
        self.shape = self.projection_ds['shape'][i]
        self.proj1 = self.projection_ds['projection'][i]
        self.composite = self.projection_ds['composite'][i]
        self.composite_plot = self.projection_ds['composite_plot'][i]
        self.nadir_proj = self.projection_ds['nadir_proj'][i]
        self.projected_set_path = self.projection_ds['projected_set_path'][i]
        self.dataset_path = self.projection_ds['general_path'][i]
        self.results_path = self.projection_ds['results_path'][i]
        self.source_img = self.photos_ds["source_img"][self.set_id]
        self.img = self.photos_ds['img'][self.set_id]
        self.tle_name = self.projection_ds['results_path'][i]
        self.res_im1 = self.projection_ds['res_im1'][i]
        self.res_im2 = self.projection_ds['res_im2'][i]
        self.res_im3 = self.projection_ds['res_im3'][i]
        self.res_im4 = self.projection_ds['res_im4'][i]
        self.res_im5 = self.projection_ds['res_im5'][i]
        self.res_final = self.projection_ds['res_final'][i]
        try:
            self.composite = float(self.composite)
        except ValueError:
            None
        times = np.linspace(self.start_time, self.end_time, 200)
        self.t = ts.tt_jd(times)

        for path in self.projection_ds["results_path"]:
            try:
                os.makedirs(path)
            except OSError:
                pass
                # print("Path exists or could't be created")

        # times = [self.start_time, self.end_time]
        # self.t = ts.tt_jd(times)
