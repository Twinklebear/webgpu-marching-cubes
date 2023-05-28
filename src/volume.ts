import {alignTo} from "./util";

export const volumes = new Map<string, string>([
    ["Fuel", "fuel_64x64x64_uint8.raw"],
    ["Neghip", "neghip_64x64x64_uint8.raw"],
    ["Hydrogen Atom", "hydrogen_atom_128x128x128_uint8.raw"],
    ["Bonsai", "bonsai_256x256x256_uint8.raw"],
    ["Foot", "foot_256x256x256_uint8.raw"],
    ["Skull", "skull_256x256x256_uint8.raw"],
    ["Aneurysm", "aneurism_256x256x256_uint8.raw"]
]);

export enum VoxelType
{
    UINT8,
    UINT16,
    UINT32,
    FLOAT32
};

export function parseVoxelType(str: string)
{
    if (str == "uint8") {
        return VoxelType.UINT8;
    }
    if (str == "uint16") {
        return VoxelType.UINT16;
    }
    if (str == "UINT32") {
        return VoxelType.UINT32;
    }
    if (str == "FLOAT32") {
        return VoxelType.FLOAT32;
    }
    throw Error(`Unsupported/invalid voxel type string ${str}`);
}

export function voxelTypeSize(ty: VoxelType)
{
    switch (ty) {
        case VoxelType.UINT8:
            return 1;
        case VoxelType.UINT16:
            return 2;
        case VoxelType.UINT32:
        case VoxelType.FLOAT32:
            return 4;
    }
}

export function voxelTypeToTextureType(ty: VoxelType)
{
    switch (ty) {
        case VoxelType.UINT8:
            // TODO: should use the non-normalized types later
            return "r8unorm";
        case VoxelType.UINT16:
            return "r16uint";
        case VoxelType.UINT32:
            return "r32uint";
        case VoxelType.FLOAT32:
            return "r32float";
    }
}

export class Volume
{
    readonly #dimensions: Array<number>;

    readonly #dataType: VoxelType;

    readonly #file: string;

    #data: Uint8Array;

    #texture: GPUTexture;

    private constructor(file: string)
    {
        let fileRegex = /(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
        let m = file.match(fileRegex);

        this.#dimensions = [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
        this.#dataType = parseVoxelType(m[5]);
        this.#file = file;
    }

    static async load(file: string, device: GPUDevice)
    {
        let volume = new Volume(file);
        await volume.fetch();
        await volume.upload(device);
        return volume;
    }

    async upload(device: GPUDevice)
    {
        this.#texture = device.createTexture({
            size: this.#dimensions,
            format: voxelTypeToTextureType(this.#dataType),
            dimension: "3d",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        // TODO: might need to chunk the upload to support very large volumes?
        // I had some note about hitting some timeout or hang issues with 512^3 in the past?
        let uploadBuf = device.createBuffer(
            {size: this.#data.byteLength, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
        new Uint8Array(uploadBuf.getMappedRange()).set(this.#data);
        uploadBuf.unmap();

        let commandEncoder = device.createCommandEncoder();

        let src = {
            buffer: uploadBuf,
            // Volumes must be aligned to 256 bytes per row, fetchVolume does this padding
            bytesPerRow: alignTo(this.#dimensions[0] * voxelTypeSize(this.#dataType), 256),
            rowsPerImage: this.#dimensions[1]
        };
        let dst = {texture: this.#texture};
        commandEncoder.copyBufferToTexture(src, dst, this.#dimensions);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    }

    get dims()
    {
        return this.#dimensions;
    }

    get dualGridDims()
    {
        return [this.#dimensions[0] - 1, this.#dimensions[1] - 1, this.#dimensions[2] - 1];
    }

    get voxelType()
    {
        return this.#dataType;
    }

    get texture()
    {
        return this.#texture;
    }

    get numVoxels()
    {
        return this.#dimensions[0] * this.#dimensions[1] * this.#dimensions[2];
    }

    get dualGridNumVoxels()
    {
        return (this.#dimensions[0] - 1) * (this.#dimensions[1] - 1) * (this.#dimensions[2] - 1);
    }

    private pad(buf: Uint8Array)
    {
        const voxelSize = voxelTypeSize(this.#dataType);

        const paddedByteDims = [
            alignTo(this.#dimensions[0] * voxelSize, 256),
            this.#dimensions[1] * voxelSize,
            this.#dimensions[2] * voxelSize
        ];
        let padded =
            new Uint8Array(paddedByteDims[0] * paddedByteDims[1] * paddedByteDims[2]);

        // Copy each row into the padded volume buffer
        const nrows = this.#dimensions[1] * this.#dimensions[2];
        for (let i = 0; i < nrows; ++i) {
            let inrow = buf.subarray(i * this.#dimensions[0] * voxelSize,
                i * this.#dimensions[0] * voxelSize + this.#dimensions[0] * voxelSize);
            padded.set(inrow, i * paddedByteDims[0]);
        }
        return padded;
    }

    private async fetch()
    {
        const voxelSize = voxelTypeSize(this.#dataType);

        const volumeSize = this.#dimensions[0] * this.#dimensions[1]
            * this.#dimensions[2] * voxelSize;

        let loadingProgressText = document.getElementById("loadingText");
        let loadingProgressBar = document.getElementById("loadingProgressBar");
        loadingProgressText.innerHTML = "Loading Volume...";
        loadingProgressBar.setAttribute("style", "width: 0%");

        let url = "https://cdn.willusher.io/demo-volumes/" + this.#file;
        try {
            let response = await fetch(url);
            let reader = response.body.getReader();

            let receivedSize = 0;
            let buf = new Uint8Array(volumeSize);
            while (true) {
                let {done, value} = await reader.read();
                if (done) {
                    break;
                }
                buf.set(value, receivedSize);
                receivedSize += value.length;
                let percentLoaded = receivedSize / volumeSize * 100;
                loadingProgressBar.setAttribute("style",
                    `width: ${percentLoaded.toFixed(2)}%`);
            }
            loadingProgressText.innerHTML = "Volume Loaded";

            // WebGPU requires that bytes per row = 256, so we need to pad volumes
            // that are smaller than this
            if ((this.#dimensions[0] * voxelSize) % 256 != 0) {
                this.#data = this.pad(buf);
            } else {
                this.#data = buf;
            }
        } catch (err) {
            loadingProgressText.innerHTML = "Error loading volume";
            throw Error(`Error loading volume data ${err}`);
        }
    }
};
