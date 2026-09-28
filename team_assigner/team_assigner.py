from sklearn.cluster import KMeans
from collections import Counter, deque
import numpy as np

class TeamAssigner:
    def __init__(self, sample_every=5, vote_window=9):
        self.team_colors = {}
        # Cada jugador se vuelve a medir cada sample_every frames y su equipo es la mayoria de las
        # ultimas vote_window mediciones: si el tracker le pasa el ID a otro jugador (ID switch),
        # el equipo se corrige en ~1 s en vez de quedarse fijo para siempre
        self.sample_every = sample_every
        self.player_votes = {}
        self.player_calls = Counter()
        self.vote_window = vote_window

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

    def get_track_color(self, frames, object_tracks, track_id, samples=15):
        # Color mediano de la camiseta de un track, muestreando algunos de sus frames
        frame_nums = [i for i, frame in enumerate(object_tracks) if track_id in frame]
        sampled = frame_nums[::max(1, len(frame_nums)//samples)]
        colors = [self.get_player_color(frames[i], object_tracks[i][track_id]["bbox"]) for i in sampled]
        return np.median(colors, axis=0)

    def team_color_ratio(self, color):
        # distancia al equipo mas cercano / distancia al otro: ~0 = camiseta de un equipo,
        # ~1 = color ajeno a ambos (arbitro, portero)
        distances = sorted(np.linalg.norm(color - self.team_colors[team]) for team in (1, 2))
        return distances[0]/distances[1]

    def fix_referees(self, frames, tracks, max_ratio=0.3):
        # YOLO a veces marca como "arbitro" a un jugador durante cientos de frames.
        # Si la camiseta de ese track es casi igual al color de un equipo (mucho mas cerca de un
        # equipo que del otro), es un jugador. Los arbitros reales quedan lejos de ambos colores.
        referee_ids = {track_id for frame in tracks["referees"] for track_id in frame}
        moved = []
        for track_id in referee_ids:
            color = self.get_track_color(frames, tracks["referees"], track_id)
            if self.team_color_ratio(color) < max_ratio:
                for i, frame in enumerate(tracks["referees"]):
                    if track_id in frame:
                        tracks["players"][i][track_id] = frame.pop(track_id)
                moved.append(int(track_id))
        return moved


    def assign_goalkeepers(self, frames, tracks, min_goalkeeper_ratio=0.25, min_color_ratio=0.5, min_players=3):
        # El portero no viste los colores de su equipo, asi que el color no sirve. Pero siempre esta
        # cerca de su porteria: si sabemos que lado defiende cada equipo, sabemos de quien es.
        # Requiere 'team' y 'position_transformed' (metros) en los tracks de jugadores.

        # 1) Porteros: tracks que YOLO vio como "goalkeeper" en parte de sus frames (no siempre acierta)
        #    y cuya camiseta NO es del color de un equipo (descarta jugadores confundidos por YOLO)
        frames_seen, frames_as_goalkeeper = Counter(), Counter()
        for frame in tracks["players"]:
            for track_id, info in frame.items():
                frames_seen[track_id] += 1
                frames_as_goalkeeper[track_id] += info.get("is_goalkeeper", False)
        goalkeepers = {track_id for track_id, n in frames_seen.items()
                       if frames_as_goalkeeper[track_id] >= min_goalkeeper_ratio*n
                       and self.team_color_ratio(self.get_track_color(frames, tracks["players"], track_id)) > min_color_ratio}

        # 2) Lado de cada equipo: en cada frame, el equipo que defiende la porteria derecha suele
        #    estar mas a la derecha que el rival. Un contragolpe invierte unos pocos frames, pero al
        #    votar con todo el video pesa poco.
        votes = []
        for frame in tracks["players"]:
            x = {1: [], 2: []}
            for track_id, info in frame.items():
                if track_id in goalkeepers or info.get("position_transformed") is None:
                    continue
                x[info["team"]].append(info["position_transformed"][0])
            if len(x[1]) >= min_players and len(x[2]) >= min_players:
                votes.append(np.mean(x[1]) > np.mean(x[2]))
        if not votes:
            return {}
        self.side_vote = float(np.mean(votes))
        self.team_right = 1 if self.side_vote >= 0.5 else 2

        # 3) Cada portero: mediana de su posicion x en todo el track -> lado -> equipo
        goalkeeper_teams = {}
        for track_id in goalkeepers:
            xs = [frame[track_id]["position_transformed"][0] for frame in tracks["players"]
                  if track_id in frame and frame[track_id].get("position_transformed") is not None]
            if not xs:
                continue
            team = self.team_right if np.median(xs) > 0 else 3 - self.team_right
            for frame in tracks["players"]:
                if track_id in frame:
                    frame[track_id]["team"] = team
                    frame[track_id]["team_color"] = self.team_colors[team]
            goalkeeper_teams[int(track_id)] = team
        return goalkeeper_teams

    def get_player_team(self,frame,player_bbox,player_id):
        votes = self.player_votes.setdefault(player_id, deque(maxlen=self.vote_window))

        if self.player_calls[player_id] % self.sample_every == 0:
            player_color = self.get_player_color(frame,player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1
            votes.append(int(team_id))
        self.player_calls[player_id] += 1

        return Counter(votes).most_common(1)[0][0]
