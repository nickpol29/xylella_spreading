










import pandas as pd







import matplotlib.pyplot as plt







import matplotlib.patches as patches







import contextily as cx











def create_plot(zoomed_in, output_filename):



    """



    Generates and saves a plot showing the dataset and selection box



    with a simple OpenStreetMap background.



    



    Args:



        zoomed_in (bool): If True, the plot is zoomed into the selection area.



        output_filename (str): The path to save the output PDF file.



    """



    try:



        full_df = pd.read_csv('sifis/fixed_test2.csv')



        filtered_df = pd.read_csv('filtered_trees_2017.csv')







        min_lat, max_lat = 40.5007, 40.5062



        min_lon, max_lon = 17.6096, 17.6200







        fig, ax = plt.subplots(figsize=(10, 10))







        # --- Set Plot Limits First ---



        box_width = max_lon - min_lon



        box_height = max_lat - min_lat



        if zoomed_in:



            ax.set_title('Area Selection for Analysis (Zoomed In)')



            margin_lat, margin_lon = box_height * 0.2, box_width * 0.2



            ax.set_xlim(min_lon - margin_lon, max_lon + margin_lon)



            ax.set_ylim(min_lat - margin_lat, max_lat + margin_lat)



            zoom_level = 17



        else:



            ax.set_title('Area Selection for Analysis (Medium View)')



            margin_lat, margin_lon = box_height * 2, box_width * 2



            ax.set_xlim(min_lon - margin_lon, max_lon + margin_lon)



            ax.set_ylim(min_lat - margin_lat, max_lat + margin_lat)



            zoom_level = 14







        # --- Plot Data on Top of Basemap ---



        ax.scatter(full_df['longitude'], full_df['latitude'], color='royalblue', alpha=0.5, s=15, zorder=3, label='Full Dataset')



        ax.scatter(filtered_df['longitude'], filtered_df['latitude'], color='green', s=30, zorder=4, label='Filtered Trees') # Changed to green







        rect = patches.Rectangle(



            (min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,



            linewidth=2, edgecolor='black', facecolor='none', linestyle='--', label='Selection Bounding Box', zorder=5



        )



        ax.add_patch(rect)







        # --- Add Annotations ---



        offset_lon = box_width * 0.1



        offset_lat = box_height * 0.1



        



        # z1 and z2 are the primary corners of the bounding box



        z1 = (min_lon, max_lat)



        z2 = (max_lon, min_lat)



        



        # a1, b1, a2, b2 are the conceptual points providing the coordinates



        a1 = (z1[0], z1[1] - offset_lat) 



        b1 = (z1[0] + offset_lon, z1[1])



        a2 = (z2[0] - offset_lon, z2[1])



        b2 = (z2[0], z2[1] + offset_lat)







        # Plot conceptual points



        ax.plot(a1[0], a1[1], 's', color='purple', markersize=8, zorder=7, label='Conceptual Input Points')



        ax.plot(b1[0], b1[1], 's', color='purple', markersize=8, zorder=7)



        ax.plot(a2[0], a2[1], 's', color='purple', markersize=8, zorder=7)



        ax.plot(b2[0], b2[1], 's', color='purple', markersize=8, zorder=7)







        # Plot bounding box corners



        ax.plot(z1[0], z1[1], 'o', color='orange', markersize=10, zorder=7, label='Bounding Box Corners') # Changed to orange



        ax.plot(z2[0], z2[1], 'o', color='orange', markersize=10, zorder=7)







        # Add labels



        text_offset = box_width * 0.04



        ax.text(z1[0] - text_offset, z1[1] + text_offset, 'z1', color='black', fontsize=12, ha='center', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))



        ax.text(z2[0] + text_offset, z2[1] - text_offset, 'z2', color='black', fontsize=12, ha='center', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))



        



        ax.text(a1[0] - text_offset, a1[1], 'a1', color='black', fontsize=12, ha='right', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))



        ax.text(b1[0], b1[1] + text_offset, 'b1', color='black', fontsize=12, va='bottom', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))



        ax.text(a2[0], a2[1] - text_offset, 'a2', color='black', fontsize=12, va='top', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))



        ax.text(b2[0] + text_offset, b2[1], 'b2', color='black', fontsize=12, ha='left', zorder=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))







        # --- Add Basemap ---



        try:



            cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, zoom=zoom_level, attribution_size=5)



        except Exception as e:



            print(f"Could not download basemap: {e}. Continuing without it.")







        ax.set_xlabel('Longitude')



        ax.set_ylabel('Latitude')



        ax.legend()



        



        plt.savefig(output_filename, format='pdf', bbox_inches='tight')



        print(f"Plot saved successfully to {output_filename}")







    except FileNotFoundError as e:



        print(f"Error: Could not find a data file. {e}")



    except Exception as e:



        print(f"An error occurred: {e}")







if __name__ == '__main__':



    create_plot(zoomed_in=False, output_filename='sifis/selection_area_zoomed_out.pdf')



    create_plot(zoomed_in=True, output_filename='sifis/selection_area_zoomed_in.pdf')






