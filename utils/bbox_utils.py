def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def scale_boxes(boxes, frame_shape, reference_size=(1920,1080)):
    # Escala cajas definidas para 1920x1080 al tamano real del frame
    sx = frame_shape[1]/reference_size[0]
    sy = frame_shape[0]/reference_size[1]
    return [(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in boxes]

def point_in_boxes(point, boxes):
    x,y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1,y1,x2,y2 in boxes)
