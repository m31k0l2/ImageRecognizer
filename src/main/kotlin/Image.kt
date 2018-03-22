import java.io.File
import java.util.regex.Pattern
import javax.imageio.ImageIO

class Image(image: File) {
    val group = getGroup(image)
    val colorsMatrix = getColorBuffers(image)
    val index = group!!.toInt()
    val netOutputs = lazy { getNetOutputs() }

    companion object {
        private fun setup(): List<MutableSet<Network>> {
            val nets = (1..20).map { mutableSetOf<Network>() }
            for (n in 0..9) {
                for (k in 0..9) {
                    NetworkIO().load("nets/nw$n$k.net")?.let {
                        nets[n].add(it)
                        nets[10+k].add(it)
                    }
                }
            }
            return nets
        }

        private val inputs = lazy { setup() }
    }

    private fun getGroup(image: File): String? {
//        val pattern = Pattern.compile("\\\\(\\d)\\\\")
        val pattern = Pattern.compile("/(\\d)/")
        val matcher = pattern.matcher(image.absolutePath)
        if (matcher.find()) {
            return matcher.group(1)
        }
        throw Exception("Нет имён соотвествующих шаблону")
    }

    private fun getColorBuffers(image: File): List<List<Double>> {
        val bufferedImage = ImageIO.read(image)
        val width = bufferedImage.width
        val height = bufferedImage.height
        val redBuffer = mutableListOf<Double>()
        for (y in 0 until height) {
            (0 until width)
                    .map { bufferedImage.getRGB(it, y) }
                    .map {
                        it shr 16 and 0xff // red
                        //                val g = rgb shr 8 and 0xff
                        //                val b = rgb and 0xff
                    }
                    .mapTo(redBuffer) {
                        it.toDouble()
                    }
        }
        return listOf(redBuffer)
    }

    private fun getNetOutputs(): List<Double> {
        val nets = inputs.value
        val t = (0..19).map { n ->
            nets[n].map {
                it.activate(this)[n % 10]
            }
        }
        return (0..19).flatMap { n ->
            nets[n].map {
                it.activate(this)[n % 10]
            }
        }
    }
}