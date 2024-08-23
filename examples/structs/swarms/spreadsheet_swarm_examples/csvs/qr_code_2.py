import qrcode

# Define the link to be converted into a QR code
link = "https://github.com/The-Swarm-Corporation/Cookbook"

# Create a QR code instance
qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR Code
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
    box_size=10,  # Size of each box in the QR code grid
    border=4,  # Thickness of the border (minimum is 4)
)

# Add the link to the QR code
qr.add_data(link)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Save the image to a file
img.save("cookbook_qr_code.png")

print("QR code generated and saved as 'cookbook_qr_code.png'")
