export const volumes = new Map([
    ["Fuel", "fuel_64x64x64_uint8.raw"],
    ["Neghip", "neghip_64x64x64_uint8.raw"],
    ["Hydrogen Atom", "hydrogen_atom_128x128x128_uint8.raw"],
    ["Bonsai", "bonsai_256x256x256_uint8.raw"],
    ["Foot", "foot_256x256x256_uint8.raw"],
    ["Skull", "skull_256x256x256_uint8.raw"],
    ["Aneurysm", "aneurism_256x256x256_uint8.raw"]
]);

export function getVolumeDimensions(file: string) {
    var fileRegex = /(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
    var m = file.match(fileRegex);
    return [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
}

export function alignTo(val: number, align: number) {
    return Math.floor((val + align - 1) / align) * align;
};

function padVolume(buf: Uint8Array, volumeDims: Array<number>) {
    const paddedVolumeDims = [alignTo(volumeDims[0], 256), volumeDims[1], volumeDims[2]];
    var padded =
        new Uint8Array(paddedVolumeDims[0] * paddedVolumeDims[1] * paddedVolumeDims[2]);
    // Copy each row into the padded volume buffer
    const nrows = volumeDims[1] * volumeDims[2];
    for (var i = 0; i < nrows; ++i) {
        var inrow = buf.subarray(i * volumeDims[0], i * volumeDims[0] + volumeDims[0]);
        padded.set(inrow, i * paddedVolumeDims[0]);
    }
    return padded;
}

export async function fetchVolume(file: string) {
    const volumeDims = getVolumeDimensions(file);
    const volumeSize = volumeDims[0] * volumeDims[1] * volumeDims[2];

    var loadingProgressText = document.getElementById("loadingText");
    var loadingProgressBar = document.getElementById("loadingProgressBar");
    loadingProgressText.innerHTML = "Loading Volume...";
    loadingProgressBar.setAttribute("style", "width: 0%");

    var url = "https://cdn.willusher.io/demo-volumes/" + file;
    try {
        var response = await fetch(url);
        var reader = response.body.getReader();

        var receivedSize = 0;
        var buf = new Uint8Array(volumeSize);
        while (true) {
            var {done, value} = await reader.read();
            if (done) {
                break;
            }
            buf.set(value, receivedSize);
            receivedSize += value.length;
            var percentLoaded = receivedSize / volumeSize * 100;
            loadingProgressBar.setAttribute("style", `width: ${percentLoaded.toFixed(2)}%`);
        }
        loadingProgressText.innerHTML = "Volume Loaded";

        // WebGPU requires that bytes per row = 256, so we need to pad volumes
        // that are smaller than this
        if (volumeDims[0] % 256 != 0) {
            return padVolume(buf, volumeDims);
        }
        return buf;
    } catch (err) {
        console.log(`Error loading volume: ${err}`);
        loadingProgressText.innerHTML = "Error loading volume";
    }
    return null;
}

export async function uploadVolume(device: GPUDevice, volumeDims: Array<number>, volumeData: Uint8Array) {
    var volumeTexture = device.createTexture({
        size: volumeDims,
        // TODO: Will need to handle other volume types here in the upload
        format: "r8unorm",
        dimension: "3d",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    // TODO: might need to chunk the upload to support very large volumes?
    // I have a note about hitting some timeout or hang issues with 512^3 in the past?
    var uploadBuf = device.createBuffer(
        {size: volumeData.length, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
    new Uint8Array(uploadBuf.getMappedRange()).set(volumeData);
    uploadBuf.unmap();

    var commandEncoder = device.createCommandEncoder();

    var src = {
        buffer: uploadBuf,
        // Volumes must be aligned to 256 bytes per row, fetchVolume does this padding
        bytesPerRow: alignTo(volumeDims[0], 256),
        rowsPerImage: volumeDims[1]
    };
    var dst = {texture: volumeTexture};
    commandEncoder.copyBufferToTexture(src, dst, volumeDims);

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    return volumeTexture;
}

export function fillSelector(selector: HTMLSelectElement, dict: Map<string, string>) {
    for (let v of dict.keys()) {
        let opt = document.createElement("option") as HTMLOptionElement;
        opt.value = v;
        opt.innerHTML = v;
        selector.appendChild(opt);
    }
}

