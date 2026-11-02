"""Coordinate transforms and image-space box geometry."""
def center_bottom_left(box, height):
    x1,y1,x2,y2 = map(float,box)
    if not (x2>x1 and y2>y1 and height>0): raise ValueError('Invalid bounding box or image height')
    return ((x1+x2)/2, height-(y1+y2)/2)
