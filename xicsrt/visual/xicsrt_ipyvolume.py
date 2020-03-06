
import ipyvolume as ipv
import numpy as np

import matplotlib

from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric

def truncate_mask(mask, max_num):
    num_mask = np.sum(mask)
    if num_mask > max_num:
        w = np.where(mask)[0]
        np.random.shuffle(w)
        mask[w[max_num:]] = False

    return mask


def figure():
    fig = ipv.figure(width=900, height=900)
    return fig

def add_rays(output, inputs):

    flag_plot_final = True
    flag_plot_unreflected = True

    color_masks = []
    color_list = []

    # This is a good pink color for un-reflected rays.
    # color_unref = [1.0, 0.5, 0.5, 0.2]

    num_optics = len(output)

    if False:
        color_masks.append(output[0]['mask'].copy())
        color_list.append((1.0, 0.0, 0.0, 0.5))

    # Color by vertical position in Plasma.
    if False:
        norm = matplotlib.colors.Normalize(0.0, 1.0)
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='gist_rainbow')

        color_masks.append((output[0]['origin'][:, 2] > 0.1))
        color_list.append(cm.to_rgba(0.0, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > 0.05) & (output[0]['origin'][:, 2] < 0.1))
        color_list.append(cm.to_rgba(1 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > 0.0) & (output[0]['origin'][:, 2] < 0.05))
        color_list.append(cm.to_rgba(2 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > -0.05) & (output[0]['origin'][:, 2] < 0))
        color_list.append(cm.to_rgba(3 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > -0.1) & (output[0]['origin'][:, 2] < -0.05))
        color_list.append(cm.to_rgba(4 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] < -0.1))
        color_list.append(cm.to_rgba(5 / 5, alpha=0.5))

    if True:
        norm = matplotlib.colors.Normalize(0.0, 1.0)
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='gist_rainbow')

        delta = 0.005
        num_wave = 5
        for ii in range(num_wave):
            wave_0 = inputs['source_input']['wavelength'] - 1 * delta * (num_wave - 1) / 2 + delta * ii
            wave_min = wave_0 - delta / 2
            wave_max = wave_0 + delta / 2
            mask_temp = (output[0]['wavelength'][:] > wave_min) & (output[0]['wavelength'][:] < wave_max)
            print('{:7.4f} {:7.4f} {:7.4f} {}'.format(wave_0, wave_min, wave_max, np.sum(mask_temp)))
            color_masks.append(mask_temp)
            color_list.append(cm.to_rgba(ii / num_wave, alpha=0.5))

    if True:
        if flag_plot_final:
            mask = output[num_optics-1]['mask'].copy()
        else:
            mask = output[1]['mask'].copy()
        num_mask = np.sum(mask)

        print('{:25s} {}'.format(
            'Graphite reflected:'
            , num_mask))

        for ii, cmask in enumerate(color_masks):
            mask_temp = mask & cmask
            num_lines = np.sum(mask_temp)
            if num_lines > 0:
                truncate_mask(mask_temp, 1000)
                num_lines = np.sum(mask_temp)

                print('{:30s} {}'.format(
                    '    Plotted:'
                    , num_lines))

                x0 = output[0]['origin'][mask_temp, 0]
                y0 = output[0]['origin'][mask_temp, 1]
                z0 = output[0]['origin'][mask_temp, 2]
                x1 = output[1]['origin'][mask_temp, 0]
                y1 = output[1]['origin'][mask_temp, 1]
                z1 = output[1]['origin'][mask_temp, 2]

                x = np.concatenate((x0, x1))
                y = np.concatenate((y0, y1))
                z = np.concatenate((z0, z1))

                lines = np.zeros((num_lines, 2), dtype=int)
                lines[:, 0] = np.arange(0, num_lines)
                lines[:, 1] = np.arange(num_lines, num_lines * 2)

                obj = ipv.plot_trisurf(x, y, z, lines=lines, color=color_list[ii])
                obj.line_material.transparent = True
                obj.line_material.linewidth = 10.0

    if True and flag_plot_unreflected:
        mask = np.invert(output[1]['mask'].copy())
        num_mask = np.sum(mask)

        truncate_mask(mask, 1000)
        num_lines = np.sum(mask)

        print('{:25s} {}'.format(
            'Graphite un-reflected:'
            , num_mask))

        if num_lines > 0:
            x0 = output[0]['origin'][mask, 0]
            y0 = output[0]['origin'][mask, 1]
            z0 = output[0]['origin'][mask, 2]
            x1 = output[1]['origin'][mask, 0]
            y1 = output[1]['origin'][mask, 1]
            z1 = output[1]['origin'][mask, 2]

            x = np.concatenate((x0, x1))
            y = np.concatenate((y0, y1))
            z = np.concatenate((z0, z1))

            lines = np.zeros((num_lines, 2), dtype=int)
            lines[:, 0] = np.arange(0, num_lines)
            lines[:, 1] = np.arange(num_lines, num_lines * 2)

            obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 1.0, 0.1])
            obj.line_material.transparent = True
            obj.line_material.linewidth = 10.0

    if True:
        if flag_plot_final:
            mask = output[num_optics-1]['mask'].copy()
        else:
            mask = output[2]['mask'].copy()
        num_mask = np.sum(mask)

        print('{:25s} {}'.format(
            'Crystal reflected:'
            , num_mask))

        for ii, cmask in enumerate(color_masks):
            mask_temp = mask & cmask
            num_lines = np.sum(mask_temp)
            if num_lines > 0:

                truncate_mask(mask_temp, 1000)
                num_lines = np.sum(mask_temp)

                x0 = output[1]['origin'][mask_temp, 0]
                y0 = output[1]['origin'][mask_temp, 1]
                z0 = output[1]['origin'][mask_temp, 2]
                x1 = output[2]['origin'][mask_temp, 0]
                y1 = output[2]['origin'][mask_temp, 1]
                z1 = output[2]['origin'][mask_temp, 2]

                x = np.concatenate((x0, x1))
                y = np.concatenate((y0, y1))
                z = np.concatenate((z0, z1))

                lines = np.zeros((num_lines, 2), dtype=int)
                lines[:, 0] = np.arange(0, num_lines)
                lines[:, 1] = np.arange(num_lines, num_lines * 2)

                obj = ipv.plot_trisurf(x, y, z, lines=lines, color=color_list[ii])
                obj.line_material.transparent = True
                obj.line_material.linewidth = 10.0

    if True and flag_plot_unreflected:
        mask = output[1]['mask'] & (~output[2]['mask'])
        num_mask = np.sum(mask)

        truncate_mask(mask, 1000)
        num_lines = np.sum(mask)

        print('{:25s} {}'.format(
            'Crystal un-reflected:'
            , num_mask))

        if num_lines > 0:
            x0 = output[1]['origin'][mask, 0]
            y0 = output[1]['origin'][mask, 1]
            z0 = output[1]['origin'][mask, 2]
            x1 = output[2]['origin'][mask, 0]
            y1 = output[2]['origin'][mask, 1]
            z1 = output[2]['origin'][mask, 2]

            x = np.concatenate((x0, x1))
            y = np.concatenate((y0, y1))
            z = np.concatenate((z0, z1))

            lines = np.zeros((num_lines, 2), dtype=int)
            lines[:, 0] = np.arange(0, num_lines)
            lines[:, 1] = np.arange(num_lines, num_lines * 2)

            obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 1.0, 0.1])
            obj.line_material.transparent = True
            obj.line_material.linewidth = 10.0

    if False:
        if flag_plot_final:
            mask = output[num_optics-1]['mask'].copy()
        else:
            mask = output[3]['mask'].copy()
        num_mask = np.sum(mask)

        print('{:25s} {}'.format(
            'Detected:'
            , num_mask))

        for ii, cmask in enumerate(color_masks):
            mask_temp = mask & cmask
            num_lines = np.sum(mask_temp)
            if num_lines > 0:

                truncate_mask(mask_temp, 1000)
                num_lines = np.sum(mask_temp)

                x0 = output[2]['origin'][mask_temp, 0]
                y0 = output[2]['origin'][mask_temp, 1]
                z0 = output[2]['origin'][mask_temp, 2]
                x1 = output[3]['origin'][mask_temp, 0]
                y1 = output[3]['origin'][mask_temp, 1]
                z1 = output[3]['origin'][mask_temp, 2]

                x = np.concatenate((x0, x1))
                y = np.concatenate((y0, y1))
                z = np.concatenate((z0, z1))

                lines = np.zeros((num_lines, 2), dtype=int)
                lines[:, 0] = np.arange(0, num_lines)
                lines[:, 1] = np.arange(num_lines, num_lines * 2)

                obj = ipv.plot_trisurf(x, y, z, lines=lines, color=color_list[ii])
                obj.line_material.transparent = True
                obj.line_material.linewidth = 10.0

def add_surf(obj):
    w = obj.param['width'] / 2.0
    h = obj.param['height'] / 2.0
    d = obj.param['depth'] / 2.0

    points = np.zeros((8, 3))
    points[0, :] = [w, h, d]
    points[1, :] = [w, -h, d]
    points[2, :] = [-w, h, d]
    points[3, :] = [-w, -h, d]

    points[4, :] = [w, h, -d]
    points[5, :] = [w, -h, -d]
    points[6, :] = [-w, h, -d]
    points[7, :] = [-w, -h, -d]

    points_ext = obj.point_to_external(points)

    x = points_ext[:, 0]
    y = points_ext[:, 1]
    z = points_ext[:, 2]

    # I am sure there is a way to automate this using a meshgrid,
    # but at the moment this is faster.
    if d > 0:
        triangles = (
            (0, 3, 1)
            , (0, 3, 2)

            , (4, 7, 5)
            , (4, 7, 6)

            , (0, 6, 2)
            , (0, 6, 4)

            , (0, 5, 1)
            , (0, 5, 4)

            , (3, 6, 2)
            , (3, 6, 7)

            , (3, 5, 1)
            , (3, 5, 7)
        )
    else:
        triangles = (
            (0, 3, 1)
            , (0, 3, 2)
        )

    ipv_obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[1.0, 1.0, 0.0, 0.2])
    ipv_obj.material.transparent = True


def add_optics(inputs):

    if True:
        crystal_center = inputs['crystal_input']['origin'] + inputs['crystal_input']['zaxis'] * inputs['crystal_input']['radius']
        x = np.array([crystal_center[0]])
        y = np.array([crystal_center[1]])
        z = np.array([crystal_center[2]])
        ipv.scatter(x, y, z, size=2, marker="sphere")

        # Plot the crystal circle.
        num = 1000
        crystal_radius = inputs['crystal_input']['radius']
        x = np.sin(np.linspace(0.0, np.pi * 2, num)) * inputs['crystal_input']['radius'] + crystal_center[0]
        y = np.cos(np.linspace(0.0, np.pi * 2, num)) * inputs['crystal_input']['radius'] + crystal_center[1]
        z = np.zeros(num) + crystal_center[2]
        lines = np.zeros((num, 2), dtype=int)
        lines[:, 0] = np.arange(num)
        lines[:, 1] = np.roll(lines[:, 0], 1)
        obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 0.0, 0.5])

        rowland_center = inputs['crystal_input']['origin'] + inputs['crystal_input']['zaxis'] * inputs['crystal_input']['radius'] / 2
        rowland_radius = crystal_radius / 2
        x = np.sin(np.linspace(0.0, np.pi * 2, num)) * rowland_radius + rowland_center[0]
        y = np.cos(np.linspace(0.0, np.pi * 2, num)) * rowland_radius + rowland_center[1]
        z = np.zeros(num) + crystal_center[2]
        lines = np.zeros((num, 2), dtype=int)
        lines[:, 0] = np.arange(num)
        lines[:, 1] = np.roll(lines[:, 0], 1)
        obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 0.0, 0.5])

    if True:
        w = inputs['crystal_input']['width'] / 2.0
        h = inputs['crystal_input']['height'] / 2.0
        cx = inputs['crystal_input']['xaxis']
        cy = np.cross(inputs['crystal_input']['xaxis'], inputs['crystal_input']['zaxis'])

        point0 = w * cx + h * cy + inputs['crystal_input']['origin']
        point1 = h * cy + inputs['crystal_input']['origin']
        point2 = -1 * w * cx + h * cy + inputs['crystal_input']['origin']

        point3 = w * cx + inputs['crystal_input']['origin']
        point4 = inputs['crystal_input']['origin']
        point5 = -1 * w * cx + inputs['crystal_input']['origin']

        point6 = w * cx - h * cy + inputs['crystal_input']['origin']
        point7 = -1 * h * cy + inputs['crystal_input']['origin']
        point8 = -1 * w * cx - h * cy + inputs['crystal_input']['origin']

        points = np.array([
            point0
            , point1
            , point2
            , point3
            , point4
            , point5
            , point6
            , point7
            , point8
        ])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # I am sure there is a way to automate this using a meshgrid,
        # but at the moment this is faster.
        triangles = (
            (4, 0, 2)
            , (4, 2, 8)
            , (4, 6, 8)
            , (4, 6, 0)
        )

        obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[0.5, 0.5, 0.5, 0.5])
        obj.material.transparent = True

    if False:
        w = inputs['graphite_input']['width'] / 2.0
        h = inputs['graphite_input']['height'] / 2.0
        cx = inputs['graphite_input']['xaxis']
        cy = np.cross(inputs['graphite_input']['xaxis'], inputs['graphite_input']['zaxis'])

        point0 = w * cx + h * cy + inputs['graphite_input']['origin']
        point1 = h * cy + inputs['graphite_input']['origin']
        point2 = -1 * w * cx + h * cy + inputs['graphite_input']['origin']

        point3 = w * cx + inputs['graphite_input']['origin']
        point4 = inputs['graphite_input']['origin']
        point5 = -1 * w * cx + inputs['graphite_input']['origin']

        point6 = w * cx - h * cy + inputs['graphite_input']['origin']
        point7 = -1 * h * cy + inputs['graphite_input']['origin']
        point8 = -1 * w * cx - h * cy + inputs['graphite_input']['origin']

        points = np.array([
            point0
            , point1
            , point2
            , point3
            , point4
            , point5
            , point6
            , point7
            , point8
        ])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # I am sure there is a way to automate this using a meshgrid,
        # but at the moment this is faster.
        triangles = (
            (4, 0, 2)
            , (4, 2, 8)
            , (4, 6, 8)
            , (4, 6, 0)
        )

        obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[0.5, 0.5, 0.5, 0.5])
        obj.material.transparent = True

    if True:
        w = inputs['detector_input']['width'] / 2.0
        h = inputs['detector_input']['height'] / 2.0
        cx = inputs['detector_input']['xaxis']
        cy = np.cross(inputs['detector_input']['xaxis'], inputs['detector_input']['zaxis'])

        point0 = w * cx + h * cy + inputs['detector_input']['origin']
        point1 = h * cy + inputs['detector_input']['origin']
        point2 = -1 * w * cx + h * cy + inputs['detector_input']['origin']

        point3 = w * cx + inputs['detector_input']['origin']
        point4 = inputs['detector_input']['origin']
        point5 = -1 * w * cx + inputs['detector_input']['origin']

        point6 = w * cx - h * cy + inputs['detector_input']['origin']
        point7 = -1 * h * cy + inputs['detector_input']['origin']
        point8 = -1 * w * cx - h * cy + inputs['detector_input']['origin']

        points = np.array([
            point0
            , point1
            , point2
            , point3
            , point4
            , point5
            , point6
            , point7
            , point8
        ])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # I am sure there is a way to automate this using a meshgrid,
        # but at the moment this is faster.
        triangles = (
            (4, 0, 2)
            , (4, 2, 8)
            , (4, 6, 8)
            , (4, 6, 0)
        )

        obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[0.5, 0.5, 0.5, 0.5])
        obj.material.transparent = True

    if False:
        w = inputs['source_input']['width'] / 2.0
        h = inputs['source_input']['height'] / 2.0
        cx = inputs['source_input']['xaxis']
        cy = np.cross(inputs['source_input']['xaxis'], inputs['source_input']['zaxis'])

        point0 = w * cx + h * cy + inputs['source_input']['origin']
        point1 = h * cy + inputs['source_input']['origin']
        point2 = -1 * w * cx + h * cy + inputs['source_input']['origin']

        point3 = w * cx + inputs['source_input']['origin']
        point4 = inputs['source_input']['origin']
        point5 = -1 * w * cx + inputs['source_input']['origin']

        point6 = w * cx - h * cy + inputs['source_input']['origin']
        point7 = -1 * h * cy + inputs['source_input']['origin']
        point8 = -1 * w * cx - h * cy + inputs['source_input']['origin']

        points = np.array([
            point0
            , point1
            , point2
            , point3
            , point4
            , point5
            , point6
            , point7
            , point8
        ])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # I am sure there is a way to automate this using a meshgrid,
        # but at the moment this is faster.
        triangles = (
            (4, 0, 2)
            , (4, 2, 8)
            , (4, 6, 8)
            , (4, 6, 0)
        )

        obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[1.0, 1.0, 0.0, 0.5])
        obj.material.transparent = True

def add_optics_volume(config):
    source = XicsrtPlasmaGeneric(config['source_input'], strict=False)
    add_surf(source)

def show():
    view = [0,0,0]
    ipv.xlim(view[1] - 0.5, view[1] + 0.5)
    ipv.ylim(view[1] - 0.5, view[1] + 0.5)
    ipv.zlim(view[2] - 0.5, view[2] + 0.5)

    #ipv.style.axes_off()
    #ipv.style.box_off()

    ipv.show()
