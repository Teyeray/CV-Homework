from PIL import Image

# 读取图片
img = Image.open('test_gray.png')

# 图像平移
def translate(img, dx, dy):
    img_trans = Image.new(img.mode, img.size, color=0)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if x+dx < width and y+dy < height:
                img_trans.putpixel((x+dx, y+dy), img.getpixel((x,y)))
    return img_trans

# 图像镜像
def mirror(img, direction):
    img_mirror = Image.new(img.mode, img.size, color=0)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if direction == 'vertical':  # 竖直镜像
                img_mirror.putpixel((x, height-y-1), img.getpixel((x,y)))
            elif direction == 'horizontal':  # 水平镜像
                img_mirror.putpixel((width-x-1, y), img.getpixel((x,y)))
    return img_mirror

# 图像旋转
def rotate(img, angle):
    img_rot = Image.new(img.mode, img.size, color=0)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            xx = int((x-width/2)*math.cos(angle) - (y-height/2)*math.sin(angle) + width/2)
            yy = int((x-width/2)*math.sin(angle) + (y-height/2)*math.cos(angle) + height/2)
            if 0 <= xx < width and 0 <= yy < height:
                img_rot.putpixel((xx,yy), img.getpixel((x,y)))
    return img_rot

# 图像复合变换
def compound_transform(img, dx, dy, mirror_direction, angle):
    img_trans = translate(img, dx, dy)
    img_mirror = mirror(img_trans, mirror_direction)
    img_rot = rotate(img_mirror, angle)
    return img_rot

# 平移图片
img_trans = translate(img, 50, 50)
#img_trans.save('trans.png')

# 镜像图片
img_mirror = mirror(img, 'horizontal')
#img_mirror.save('mirror.png')

# 旋转图片
import math
img_rot = rotate(img, math.pi/6)
#img_rot.save('rotate.png')

# 复合变换
img_compound = compound_transform(img, 50, 50, 'horizontal', math.pi/6)
#img_compound.save('compound.png')


#显示图片
img_trans.show()
img_mirror.show()
img_compound.show()
