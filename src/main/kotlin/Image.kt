import java.io.File
import java.util.regex.Pattern
import javax.imageio.ImageIO

class Image(image: File) {
    val group = getGroup(image)
    val colorsMatrix = getColorBuffers(image)
    val index = group!!.toInt()

    private fun getGroup(image: File): String? {
        val pattern = Pattern.compile("\\\\(\\d)\\\\")
//        val pattern = Pattern.compile("/(\\d)/")
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
}