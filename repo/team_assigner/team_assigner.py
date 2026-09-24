from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

class TeamAssigner:
    def __init__(self, votes_per_player=10):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_votes = {}
        self.votes_per_player = votes_per_player

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frames, player_tracks, num_frames=10):
        # Juntar colores de camiseta de varios frames (no solo el primero) para un K-Means mas estable
        frame_nums = np.linspace(0, len(frames)-1, num_frames, dtype=int)

        player_colors = []
        for frame_num in frame_nums:
            for _, player_detection in player_tracks[frame_num].items():
                bbox = player_detection["bbox"]
                player_color =  self.get_player_color(frames[frame_num],bbox)
                player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        # Votar durante las primeras apariciones del jugador y fijar el equipo por mayoria
        votes = self.player_votes.setdefault(player_id, Counter())
        votes[team_id] += 1
        team_id = votes.most_common(1)[0][0]
        if sum(votes.values()) >= self.votes_per_player:
            self.player_team_dict[player_id] = team_id

        return team_id
