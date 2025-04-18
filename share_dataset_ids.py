import qai_hub as hub

tflite_normal = [
	'dv74k8ew2', 'dv910vo82', 'dq9krm657', 'd82ndxp57', 'dv9518om2',
	'dd9ppq5n9', 'dz7z43qr9', 'd67jwmon2', 'd67oxpon7', 'dp7lgeow2',
	'dk7gkz4o2', 'dv74k8qw2', 'dq9krmo57', 'dp70nz3l9', 'd82ndxo57',
	'd09y13p39', 'dv95185m2', 'dd9ppqon9', 'dn7xzrx59', 'dj7d0em89',
	'dz7z43vr9', 'dx9e8e0p9', 'd67jwm4n2', 'dw9v84vj7', 'd693mv6l7',
	'dp7lge3w2', 'dk7gkzmo2', 'dz2r543o7', 'dr9wme332', 'dv74k83w2',
	'dq9krm357', 'd678rge62', 'd09y13v39', 'dv95186m2', 'dr2qq53l2',
	'dj7d0en89', 'dr2qq51o2', 'dn7xzrnv9', 'dz7z43869', 'dx9e8e149',
	'dw9v84507', 'd67oxpmq7', 'd693mvwp7', 'dw264zde9', 'dk7gkzx02',
	'dr9wmeyk2', 'dv910vwe2', 'd678rgpy2', 'd09y130m9', 'dd9ppqew9',
]

tflite_quantized = [
	'dz7z4dmz9', 'dx9e8lok9', 'd67jwpq42', 'dw9v8gy57', 'd67ox08w7',
	'dw2645359', 'dp7lg5yr2', 'dk7gkldg2', 'dr9wmkne2', 'dv9101zg2',
	'do7mlynv9', 'd82ndnn17', 'd09y1ooe9', 'dr2qqkk82', 'dz7z4ddz9',
	'd67jwpp42', 'dw2645559', 'dk7gkllg2', 'dz2r5kkp7', 'do7mlyyv9',
	'd678rnnw2', 'dr2qqkm82', 'dn7xz6lz9', 'dj7d0zqy9', 'dd9pp8ro9',
	'dw9v8gmm7', 'dp7lg5r42', 'dz2r5kvl7', 'dv74kpr02', 'dq9krqk87',
	'do7mlyem9', 'dp70njld9', 'd82ndn6q7', 'dv951qjn2', 'dd9pp8zo9',
	'dz7z4dy59', 'dx9e8lzo9', 'd67jwplp2', 'dw9v8gxm7', 'd693mxey7',
	'dp7lg5k42', 'dr9wmkpw2', 'dv9101qn2', 'do7mlyqm9', 'd09y1o619',
	'dd9pp8do9', 'dz7z4dl59', 'd693mxyy7', 'dk7gklvy2', 'dr9wmkgw2'
]

onnx_normal = [
	'dj7d0qwq9', 'dz7z4jny9', 'dx9e85mv9', 'dw9v8m6z7', 'd67oxrvl7',
	'd693mj007', 'dw264vqg9', 'dp7lgrv12', 'dk7gkn1e2', 'dz2r5veg7',
	'dr9wm6wo2', 'dv74krjz2', 'dv9103yx2', 'dq9krg1d7', 'dp70nmko9',
	'd09y1jmv9', 'dd9pprgm9', 'dj7d0q4q9', 'dz7z4j0y9', 'dw9v8mpz7',
	'd67oxrzl7', 'dz2r5v8g7', 'dv74kr0z2', 'do7ml6zl9', 'd82ndr4o7',
	'dr2qqmgv2', 'dn7xzly69', 'dz7z4joy9', 'd693mjn07', 'dw264vkg9',
	'dz2r56jg7', 'dr9wmp8o2', 'do7mle1l9', 'd678r4vm2', 'd82nd6wo7',
	'dd9ppz6m9', 'dn7xzqk69', 'dj7d03lp9', 'dz7z4yrw9', 'dw9v8xnq7',
	'dw264wj69', 'dz2r56g07', 'dv74kloy2', 'do7mlek39', 'dp70nlg09',
	'dv951jvz2', 'dr2qq6r62', 'dd9ppzwd9', 'dn7xzqjr9', 'dj7d03xp9'
]

onnx_quantized = [
	'dp70njzd9', 'dz7z4d359', 'd67jwpmp2', 'd693mxvy7', 'dk7gklzy2',
	'do7mly4m9', 'd82ndnvq7', 'dr2qqk0y2', 'dr2qqk0v2', 'dj7d0zpq9',
	'd67ox0el7', 'dp7lg5412', 'dz2r5k1g7', 'dv9101gx2', 'do7mlygl9',
	'd82ndnlo7', 'dr2qqklv2', 'dn7xz6369', 'dw9v8glz7', 'd693mxd07',
	'dp7lg5612', 'dz2r5klg7', 'dv74kpnz2', 'dv91015x2', 'dq9krqvd7',
	'd82ndnmo7', 'd09y1ozv9', 'dr2qqk4v2', 'dd9pp8nm9', 'dz7z4dny9',
	'd67jwpgm2', 'dp7lg5v12', 'dz2r5keg7', 'dq9krq1d7', 'd678rnkm2',
	'dp70njko9', 'dr2qqkev2', 'dj7d0z4q9', 'd67jwpzm2', 'dk7gklpe2',
	'do7mlyzl9', 'd82ndn4o7', 'dv951qge2', 'dj7d0z8q9', 'd67jwpez2',
	'd67oxrgg7', 'dw264vl69', 'dv74kr4y2', 'do7ml6139', 'dp70nm509'
]

def add_sharing(dataset_ids):
	for dataset_id in dataset_ids:
		dataset = hub.get_dataset(dataset_id)
		dataset.modify_sharing(["andre150@csusm.edu", "keoph002@csusm.edu", "janov005@csusm.edu"],[])
		print(f"Shared '{dataset.name}' with: {dataset.get_sharing()}")

print("----------------------------- tflite_normal -----------------------------")
add_sharing(tflite_normal)
print("----------------------------- tflite_quantized -----------------------------")
add_sharing(tflite_quantized)
print("----------------------------- onnx_normal -----------------------------")
add_sharing(onnx_normal)
print("----------------------------- onnx_quantized -----------------------------")
add_sharing(onnx_quantized)
