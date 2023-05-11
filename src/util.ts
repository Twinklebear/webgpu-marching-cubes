export function alignTo(val: number, align: number)
{
    return Math.floor((val + align - 1) / align) * align;
};

// Compute the shader and print any error log
export async function compileShader(device: GPUDevice, src: string, debugLabel?: string)
{
    let shaderModule = device.createShaderModule({code: src});
    let compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        let hadError = false;
        if (debugLabel) {
            console.log(`Shader compilation log for ${debugLabel}:`);
        } else {
            console.log(`Shader compilation log:`);
        }
        for (let i = 0; i < compilationInfo.messages.length; ++i) {
            let msg = compilationInfo.messages[i];
            console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
            hadError = hadError || msg.type == "error";
        }
        if (hadError) {
            console.log("Shader failed to compile");
            throw Error("Shader failed to compile");
        }
    }
    return shaderModule;
}

export function fillSelector(selector: HTMLSelectElement, dict: Map<string, string>)
{
    for (let v of dict.keys()) {
        let opt = document.createElement("option") as HTMLOptionElement;
        opt.value = v;
        opt.innerHTML = v;
        selector.appendChild(opt);
    }
}
